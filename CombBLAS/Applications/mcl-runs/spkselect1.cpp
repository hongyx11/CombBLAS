/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#include "SpParMat.h"
#include "ParFriends.h"
#include "Operations.h"
#include "FileHeader.h"
extern "C" {
#include "mmio.h"
}
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <fstream>
#include <algorithm>
#include <stdexcept>
using namespace std;

/**
  * If every processor has a distinct triples file such as {A_0, A_1, A_2,... A_p} for p processors
 **/
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (ifstream & input, MPI_Comm & world)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}
	commGrid.reset(new CommGrid(world, 0, 0));
	input >> (*spSeq);
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, MPI_Comm & world): spSeq(myseq)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	commGrid.reset(new CommGrid(world, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, shared_ptr<CommGrid> grid): spSeq(myseq)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	commGrid = grid;
}	

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (shared_ptr<CommGrid> grid)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	spSeq = new DER();
	commGrid = grid;
}

//! Deprecated. Don't call the default constructor
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat ()
{
	SpParHelper::Print("COMBBLAS Warning: It is dangerous to create (matrix) objects without specifying the communicator, are you sure you want to create this object in MPI_COMM_WORLD?\n");
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	spSeq = new DER();
	commGrid.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
}

/**
* If there is a single file read by the master process only, use this and then call ReadDistribute()
**/
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (MPI_Comm world)
{
    
    assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
    spSeq = new DER();
    commGrid.reset(new CommGrid(world, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::~SpParMat ()
{
	if(spSeq != NULL) delete spSeq;
}

template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::FreeMemory ()
{
	if(spSeq != NULL) delete spSeq;
	spSeq = NULL;
}


/**
 * Private function to guide Select2 communication and avoid code duplication due to loop ends
 * @param[int, out] klimits {per column k limit gets updated for the next iteration}
 * @param[out] converged {items to remove from actcolsmap at next iteration{
 **/
template <class IT, class NT, class DER>
template <typename VT, typename GIT>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::TopKGather(vector<NT> & all_medians, vector<IT> & nnz_per_col, int & thischunk, int & chunksize,
                                     const vector<NT> & activemedians, const vector<IT> & activennzperc, int itersuntil,
                                     vector< vector<NT> > & localmat, const vector<IT> & actcolsmap, vector<IT> & klimits,
                                     vector<IT> & toretain, vector<vector<pair<IT,NT>>> & tmppair, IT coffset, const FullyDistVec<GIT,VT> & rvec) const
{
    int rankincol = commGrid->GetRankInProcCol();
    int colneighs = commGrid->GetGridRows();
    int nprocs = commGrid->GetSize();
    vector<double> finalWeightedMedians(thischunk, 0.0);
    
    MPI_Gather(activemedians.data() + itersuntil*chunksize, thischunk, MPIType<NT>(), all_medians.data(), thischunk, MPIType<NT>(), 0, commGrid->GetColWorld());
    MPI_Gather(activennzperc.data() + itersuntil*chunksize, thischunk, MPIType<IT>(), nnz_per_col.data(), thischunk, MPIType<IT>(), 0, commGrid->GetColWorld());

    if(rankincol == 0)
    {
        vector<double> columnCounts(thischunk, 0.0);
        vector< pair<NT, double> > mediansNweights(colneighs);  // (median,weight) pairs    [to be reused at each iteration]
        
        for(int j = 0; j < thischunk; ++j)  // for each column
        {
            for(int k = 0; k<colneighs; ++k)
            {
                size_t fetchindex = k*thischunk+j;
                columnCounts[j] += static_cast<double>(nnz_per_col[fetchindex]);
            }
            for(int k = 0; k<colneighs; ++k)
            {
                size_t fetchindex = k*thischunk+j;
                mediansNweights[k] = make_pair(all_medians[fetchindex], static_cast<double>(nnz_per_col[fetchindex]) / columnCounts[j]);
            }
            sort(mediansNweights.begin(), mediansNweights.end());   // sort by median
            
            double sumofweights = 0;
            int k = 0;
            while( k<colneighs && sumofweights < 0.5)
            {
                sumofweights += mediansNweights[k++].second;
            }
            finalWeightedMedians[j] = mediansNweights[k-1].first;
        }
    }
    MPI_Bcast(finalWeightedMedians.data(), thischunk, MPIType<double>(), 0, commGrid->GetColWorld());
    
    vector<IT> larger(thischunk, 0);
    vector<IT> smaller(thischunk, 0);
    vector<IT> equal(thischunk, 0);

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int j = 0; j < thischunk; ++j)  // for each active column
    {
        size_t fetchindex = actcolsmap[j+itersuntil*chunksize];        
        for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
        {
            // count those above/below/equal to the median
            if(localmat[fetchindex][k] > finalWeightedMedians[j])
                larger[j]++;
            else if(localmat[fetchindex][k] < finalWeightedMedians[j])
                smaller[j]++;
            else
                equal[j]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, larger.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    MPI_Allreduce(MPI_IN_PLACE, smaller.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    MPI_Allreduce(MPI_IN_PLACE, equal.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    
    int numThreads = 1;	// default case
#ifdef THREADED
    omp_lock_t lock[nprocs];    // a lock per recipient
    for (int i=0; i<nprocs; i++)
        omp_init_lock(&(lock[i]));
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    
    vector < vector<IT> > perthread2retain(numThreads);
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int j = 0; j < thischunk; ++j)  // for each active column
    {
#ifdef THREADED
        int myThread = omp_get_thread_num();
#else
        int myThread = 0;
#endif
        
        // both clmapindex and fetchindex are unique for a given j (hence not shared among threads)
        size_t clmapindex = j+itersuntil*chunksize;     // klimits is of the same length as actcolsmap
        size_t fetchindex = actcolsmap[clmapindex];     // localmat can only be dereferenced using the original indices.
        
        // these following if/else checks are the same (because klimits/large/equal vectors are mirrored) on every processor along ColWorld
        if(klimits[clmapindex] <= larger[j]) // the entries larger than Weighted-Median are plentiful, we can discard all the smaller/equal guys
        {
            vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] > finalWeightedMedians[j])  // keep only the large guys (even equal guys go)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            perthread2retain[myThread].push_back(clmapindex);    // items to retain in actcolsmap
        }
        else if (klimits[clmapindex] > larger[j] + equal[j]) // the elements that are either larger or equal-to are surely keepers, no need to reprocess them
        {
            vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] < finalWeightedMedians[j])  // keep only the small guys (even equal guys go)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            
            klimits[clmapindex] -= (larger[j] + equal[j]);   // update the k limit for this column only
            perthread2retain[myThread].push_back(clmapindex);    // items to retain in actcolsmap
        }
        else  // larger[j] < klimits[clmapindex] &&  klimits[clmapindex] <= larger[j] + equal[j]
        {
            vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] >= finalWeightedMedians[j])  // keep the larger and equal to guys (might exceed k-limit but that's fine according to MCL)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            
            // We found it: the kth largest element in column (coffset + fetchindex) is finalWeightedMedians[j]
            // But everyone in the same processor column has the information, only one of them should send it
            IT n_perproc = getlocalcols() / colneighs;  // find a typical processor's share
            int assigned = std::max(static_cast<int>(fetchindex/n_perproc), colneighs-1);
            if( assigned == rankincol)
            {
                IT locid;
                int owner = rvec.Owner(coffset + fetchindex, locid);
                
            #ifdef THREADED
                omp_set_lock(&(lock[owner]));
            #endif
                tmppair[owner].emplace_back(make_pair(locid, finalWeightedMedians[j]));
            #ifdef THREADED
                omp_unset_lock(&(lock[owner]));
            #endif
            }
        } // end_else
    } // end_for
    // ------ concatenate toretain "indices" processed by threads ------
    vector<IT> tdisp(numThreads+1);
    tdisp[0] = 0;
    for(int i=0; i<numThreads; ++i)
    {
        tdisp[i+1] = tdisp[i] + perthread2retain[i].size();
    }
    toretain.resize(tdisp[numThreads]);
    
#pragma omp parallel for
    for(int i=0; i< numThreads; i++)
    {
        std::copy(perthread2retain[i].data() , perthread2retain[i].data()+ perthread2retain[i].size(), toretain.data() + tdisp[i]);
    }
    
#ifdef THREADED
    for (int i=0; i<nprocs; i++)    // destroy the locks
        omp_destroy_lock(&(lock[i]));
#endif
}


//! identify the k-th maximum element in each column of a matrix
//! if the number of nonzeros in a column is less then k, return the minimum among the entries of that column (including the implicit zero)
//! This is an efficient implementation of the Saukas/Song algorithm
//! http://www.ime.usp.br/~einar/select/INDEX.HTM
//! Preferred for large k values
template <class IT, class NT, class DER>
template <typename VT, typename GIT>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect2(FullyDistVec<GIT,VT> & rvec, IT k_limit) const
{
    
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    IT locm = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)
    vector< vector<NT> > localmat(locm);    // some sort of minimal local copy of matrix
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
        {
            localmat[colit.colid()].push_back(nzit.value());
        }
    }
    
    vector<IT> nnzperc(locm); // one per column
    int rankincol = commGrid->GetRankInProcCol();
    int rankinrow = commGrid->GetRankInProcRow();
    int rowneighs = commGrid->GetGridCols();	// get # of processors on the row
    int colneighs = commGrid->GetGridRows();
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    
    for(IT i=0; i<locm; i++)
        nnzperc[i] = localmat[i].size();
    
    vector<IT> percsum(locm, 0);
    MPI_Allreduce(nnzperc.data(), percsum.data(), locm, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());

    nnzperc.resize(0);
    nnzperc.shrink_to_fit();
    
    int64_t activecols = std::count_if(percsum.begin(), percsum.end(), [k_limit](IT i){ return i > k_limit;});
    int64_t activennz = std::accumulate(percsum.begin(), percsum.end(), (int64_t) 0);
    
    int64_t totactcols, totactnnzs;
    MPI_Allreduce(&activecols, &totactcols, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
    if(myrank == 0)   cout << "Number of initial active columns are " << totactcols << endl;

    MPI_Allreduce(&activennz, &totactnnzs, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
    if(myrank == 0)   cout << "Number of initial nonzeros are " << totactnnzs << endl;
    
    Reduce(rvec, Column, minimum<NT>(), static_cast<NT>(0));    // get the vector ready, this should also set the glen of rvec correctly
    
    
#ifdef COMBBLAS_DEBUG
    PrintInfo();
    rvec.PrintInfo("rvector");
#endif
    
    if(totactcols == 0)
    {
        ostringstream ss;
        ss << "TopK: k_limit (" << k_limit <<")" << " >= maxNnzInColumn. Returning the result of Reduce(Column, minimum<NT>()) instead..." << endl;
        SpParHelper::Print(ss.str());
        return false;   // no prune needed
    }
   
    
    vector<IT> actcolsmap(activecols);  // the map that gives the original index of that active column (this map will shrink over iterations)
    for (IT i=0, j=0; i< locm; ++i) {
        if(percsum[i] > k_limit)
            actcolsmap[j++] = i;
    }
    
    vector<NT> all_medians;
    vector<IT> nnz_per_col;
    vector<IT> klimits(activecols, k_limit); // is distributed management of this vector needed?
    int activecols_lowerbound = 10*colneighs;
    
    
    IT * locncols = new IT[rowneighs];
    locncols[rankinrow] = locm;
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetRowWorld());
    IT coffset = accumulate(locncols, locncols+rankinrow, static_cast<IT>(0));
    delete [] locncols;
    
    /* Create/allocate variables for vector assignment */
    MPI_Datatype MPI_pair;
    MPI_Type_contiguous(sizeof(pair<IT,NT>), MPI_CHAR, &MPI_pair);
    MPI_Type_commit(&MPI_pair);
    
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    while(totactcols > 0)
    {
        int chunksize, iterations, lastchunk;
        if(activecols > activecols_lowerbound)
        {
            // two reasons for chunking:
            // (1) keep memory limited to activecols (<= n/sqrt(p))
            // (2) avoid overflow in sentcount
            chunksize = static_cast<int>(activecols/colneighs); // invariant chunksize >= 10 (by activecols_lowerbound)
            iterations = std::max(static_cast<int>(activecols/chunksize), 1);
            lastchunk = activecols - (iterations-1)*chunksize; // lastchunk >= chunksize by construction
        }
        else
        {
            chunksize = activecols;
            iterations = 1;
            lastchunk = activecols;
        }
        vector<NT> activemedians(activecols);   // one per "active" column
        vector<IT> activennzperc(activecols);
   
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i< activecols; ++i) // recompute the medians and nnzperc
        {
            size_t orgindex = actcolsmap[i];	// assert: no two threads will share the same "orgindex"
            if(localmat[orgindex].empty())
            {
                activemedians[i] = (NT) 0;
                activennzperc[i] = 0;
            }
            else
            {
                // this actually *sorts* increasing but doesn't matter as long we solely care about the median as opposed to a general nth element
                auto entriesincol(localmat[orgindex]);   // create a temporary vector as nth_element modifies the vector
                std::nth_element(entriesincol.begin(), entriesincol.begin() + entriesincol.size()/2, entriesincol.end());
                activemedians[i] = entriesincol[entriesincol.size()/2];
                activennzperc[i] = entriesincol.size();
            }
        }
        
        percsum.resize(activecols, 0);
        MPI_Allreduce(activennzperc.data(), percsum.data(), activecols, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
        activennz = std::accumulate(percsum.begin(), percsum.end(), (int64_t) 0);
        
#ifdef COMBBLAS_DEBUG
        MPI_Allreduce(&activennz, &totactnnzs, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
        if(myrank == 0)   cout << "Number of active nonzeros are " << totactnnzs << endl;
#endif
        
        vector<IT> toretain;
        if(rankincol == 0)
        {
            all_medians.resize(lastchunk*colneighs);
            nnz_per_col.resize(lastchunk*colneighs);
        }
        vector< vector< pair<IT,NT> > > tmppair(nprocs);
        for(int i=0; i< iterations-1; ++i)  // this loop should not be parallelized if we want to keep storage small
        {
            TopKGather(all_medians, nnz_per_col, chunksize, chunksize, activemedians, activennzperc, i, localmat, actcolsmap, klimits, toretain, tmppair, coffset, rvec);
        }
        TopKGather(all_medians, nnz_per_col, lastchunk, chunksize, activemedians, activennzperc, iterations-1, localmat, actcolsmap, klimits, toretain, tmppair, coffset, rvec);
        
        /* Set the newly found vector entries */
        IT totsend = 0;
        for(IT i=0; i<nprocs; ++i)
        {
            sendcnt[i] = tmppair[i].size();
            totsend += tmppair[i].size();
        }
        
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());
        
        partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
        partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
        IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
        
        pair<IT,NT> * sendpair = new pair<IT,NT>[totsend];
        for(int i=0; i<nprocs; ++i)
        {
            copy(tmppair[i].begin(), tmppair[i].end(), sendpair+sdispls[i]);
            vector< pair<IT,NT> >().swap(tmppair[i]);	// clear memory
        }
        vector< pair<IT,NT> > recvpair(totrecv);
        MPI_Alltoallv(sendpair, sendcnt, sdispls, MPI_pair, recvpair.data(), recvcnt, rdispls, MPI_pair, commGrid->GetWorld());
        delete [] sendpair;

        IT updated = 0;
        for(auto & update : recvpair )    // Now, write these to rvec
        {
            updated++;
            rvec.arr[update.first] =  update.second;
        }
#ifdef COMBBLAS_DEBUG
        MPI_Allreduce(MPI_IN_PLACE, &updated, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
        if(myrank  == 0) cout << "Total vector entries updated " << updated << endl;
#endif

        /* End of setting up the newly found vector entries */
        
        vector<IT> newactivecols(toretain.size());
        vector<IT> newklimits(toretain.size());
        IT newindex = 0;
        for(auto & retind : toretain )
        {
            newactivecols[newindex] = actcolsmap[retind];
            newklimits[newindex++] = klimits[retind];
        }
        actcolsmap.swap(newactivecols);
        klimits.swap(newklimits);
        activecols = actcolsmap.size();
        
        MPI_Allreduce(&activecols, &totactcols, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
#ifdef COMBBLAS_DEBUG
        if(myrank  == 0) cout << "Number of active columns are " << totactcols << endl;
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
    MPI_Type_free(&MPI_pair);
    MPI_Barrier(MPI_COMM_WORLD);
    
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)   cout << "Exiting kselect2"<< endl;
#endif
    return true;    // prune needed
}



template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::Dump(string filename) const
{
	MPI_Comm World = commGrid->GetWorld();
	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
		
	MPI_File thefile;
    char * _fn = const_cast<char*>(filename.c_str());
	MPI_File_open(World, _fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);

	int rankinrow = commGrid->GetRankInProcRow();
	int rankincol = commGrid->GetRankInProcCol();
	int rowneighs = commGrid->GetGridCols();	// get # of processors on the row
	int colneighs = commGrid->GetGridRows();

	IT * colcnts = new IT[rowneighs];
	IT * rowcnts = new IT[colneighs];
	rowcnts[rankincol] = getlocalrows();
	colcnts[rankinrow] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), colcnts, 1, MPIType<IT>(), commGrid->GetRowWorld());
	IT coloffset = accumulate(colcnts, colcnts+rankinrow, static_cast<IT>(0));

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), rowcnts, 1, MPIType<IT>(), commGrid->GetColWorld());	
	IT rowoffset = accumulate(rowcnts, rowcnts+rankincol, static_cast<IT>(0));
	DeleteAll(colcnts, rowcnts);

	IT * prelens = new IT[nprocs];
	prelens[rank] = 2*getlocalnnz();
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT lengthuntil = accumulate(prelens, prelens+rank, static_cast<IT>(0));

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
	MPI_Offset disp = lengthuntil * sizeof(uint32_t);
	char native[] = "native";
	MPI_File_set_view(thefile, disp, MPI_UNSIGNED, MPI_UNSIGNED, native, MPI_INFO_NULL); // AL: the second-to-last argument is a non-const char* (looks like poor MPI standardization, the C++ bindings list it as const), C++ string literals MUST be const (especially in c++11).
	uint32_t * gen_edges = new uint32_t[prelens[rank]];
	
	IT k = 0;
	for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
	{
		for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
		{
			gen_edges[k++] = (uint32_t) (nzit.rowid() + rowoffset);
			gen_edges[k++] = (uint32_t) (colit.colid() +  coloffset);
		}
	}
	assert(k == prelens[rank]);
	MPI_File_write(thefile, gen_edges, prelens[rank], MPI_UNSIGNED, NULL);	
	MPI_File_close(&thefile);

	delete [] prelens;
	delete [] gen_edges;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (const SpParMat< IT,NT,DER > & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new DER(*(rhs.spSeq));  	// Deep copy of local block

	commGrid =  rhs.commGrid;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER > & SpParMat< IT,NT,DER >::operator=(const SpParMat< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		//! Check agains NULL is probably unneccessary, delete won't fail on NULL
		//! But useful in the presence of a user defined "operator delete" which fails to check NULL
		if(spSeq != NULL) delete spSeq;
		if(rhs.spSeq != NULL)	
			spSeq = new DER(*(rhs.spSeq));  // Deep copy of local block
	
		commGrid = rhs.commGrid;
	}
	return *this;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER > & SpParMat< IT,NT,DER >::operator+=(const SpParMat< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		if(*commGrid == *rhs.commGrid)	
		{
			(*spSeq) += (*(rhs.spSeq));
		}
		else
		{
			cout << "Grids are not comparable for parallel addition (A+B)" << endl; 
		}
	}
	else
	{
		cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;	
}

template <class IT, class NT, class DER>
float SpParMat< IT,NT,DER >::LoadImbalance() const
{
	IT totnnz = getnnz();	// collective call
	IT maxnnz = 0;    
	IT localnnz = (IT) spSeq->getnnz();
	MPI_Allreduce( &localnnz, &maxnnz, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
	if(totnnz == 0) return 1;
 	return static_cast<float>((commGrid->GetSize() * maxnnz)) / static_cast<float>(totnnz);  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnnz() const
{
	IT totalnnz = 0;    
	IT localnnz = spSeq->getnnz();
	MPI_Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
 	return totalnnz;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnrow() const
{
	IT totalrows = 0;
	IT localrows = spSeq->getnrow();    
	MPI_Allreduce( &localrows, &totalrows, 1, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
 	return totalrows;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getncol() const
{
	IT totalcols = 0;
	IT localcols = spSeq->getncol();    
	MPI_Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI_SUM, commGrid->GetRowWorld());
 	return totalcols;  
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::DimApply(Dim dim, const FullyDistVec<IT, NT>& x, _BinaryOperation __binary_op)
{

	if(!(*commGrid == *(x.commGrid))) 		
	{
		cout << "Grids are not comparable for SpParMat::DimApply" << endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();
	switch(dim)
	{
		case Column:	// scale each column
		{
			int xsize = (int) x.LocArrSize();
			int trxsize = 0;
			int diagneigh = x.commGrid->GetComplementRank();
			MPI_Status status;
			MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	
			NT * trxnums = new NT[trxsize];
			MPI_Sendrecv(const_cast<NT*>(SpHelper::p2a(x.arr)), xsize, MPIType<NT>(), diagneigh, TRX, trxnums, trxsize, MPIType<NT>(), diagneigh, TRX, World, &status);

			int colneighs, colrank;
			MPI_Comm_size(ColWorld, &colneighs);
			MPI_Comm_rank(ColWorld, &colrank);
			int * colsize = new int[colneighs];
			colsize[colrank] = trxsize;
		
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, ColWorld);	
			int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
			int accsize = std::accumulate(colsize, colsize+colneighs, 0);
			NT * scaler = new NT[accsize];

			MPI_Allgatherv(trxnums, trxsize, MPIType<NT>(), scaler, colsize, dpls, MPIType<NT>(), ColWorld);
			DeleteAll(trxnums,colsize, dpls);

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() = __binary_op(nzit.value(), scaler[colit.colid()]);
				}
			}
			delete [] scaler;
			break;
		}
		case Row:
		{
			int xsize = (int) x.LocArrSize();
			int rowneighs, rowrank;
			MPI_Comm_size(RowWorld, &rowneighs);
			MPI_Comm_rank(RowWorld, &rowrank);
			int * rowsize = new int[rowneighs];
			rowsize[rowrank] = xsize;
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rowsize, 1, MPI_INT, RowWorld);
			int * dpls = new int[rowneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(rowsize, rowsize+rowneighs-1, dpls+1);
			int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
			NT * scaler = new NT[accsize];

			MPI_Allgatherv(const_cast<NT*>(SpHelper::p2a(x.arr)), xsize, MPIType<NT>(), scaler, rowsize, dpls, MPIType<NT>(), RowWorld);
			DeleteAll(rowsize, dpls);

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() = __binary_op(nzit.value(), scaler[nzit.rowid()]);
				}
			}
			delete [] scaler;			
			break;
		}
		default:
		{
			cout << "Unknown scaling dimension, returning..." << endl;
			break;
		}
	}
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename _UnaryOperation >	
FullyDistVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const
{
    IT length;
    switch(dim)
    {
        case Column:
        {
            length = getncol();
            break;
        }
        case Row:
        {
            length = getnrow();
            break;
        }
        default:
        {
            cout << "Unknown reduction dimension, returning empty vector" << endl;
            break;
        }
    }
	FullyDistVec<IT,NT> parvec(commGrid, length, id);
	Reduce(parvec, dim, __binary_op, id, __unary_op);			
	return parvec;
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
FullyDistVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const
{
    IT length;
    switch(dim)
    {
        case Column:
        {
            length = getncol();
            break;
        }
        case Row:
        {
            length = getnrow();
            break;
        }
        default:
        {
            cout << "Unknown reduction dimension, returning empty vector" << endl;
            break;
        }
    }
	FullyDistVec<IT,NT> parvec(commGrid, length, id);
	Reduce(parvec, dim, __binary_op, id, myidentity<NT>()); // myidentity<NT>() is a no-op function
	return parvec;
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const
{
	Reduce(rvec, dim, __binary_op, id, myidentity<NT>() );				
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector	
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const
{
    Reduce(rvec, dim, __binary_op, id, __unary_op, MPIOp<_BinaryOperation, VT>::op() );
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, MPI_Op mympiop) const
{
	if(*rvec.commGrid != *commGrid)
	{
		SpParHelper::Print("Grids are not comparable, SpParMat::Reduce() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
	}
	switch(dim)
	{
		case Column:	// pack along the columns, result is a vector of size n
		{
			// We can't use rvec's distribution (rows first, columns later) here
            // ABAB (2017): Why not? I can't think of a counter example where it wouldn't work
            IT n_thiscol = getlocalcols();   // length assigned to this processor column
			int colneighs = commGrid->GetGridRows();	// including oneself
            int colrank = commGrid->GetRankInProcCol();

			GIT * loclens = new GIT[colneighs];
			GIT * lensums = new GIT[colneighs+1]();	// begin/end points of local lengths

            GIT n_perproc = n_thiscol / colneighs;    // length on a typical processor
            if(colrank == colneighs-1)
                loclens[colrank] = n_thiscol - (n_perproc*colrank);
            else
                loclens[colrank] = n_perproc;

			MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetColWorld());
			partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits

			vector<VT> trarr;
			typename DER::SpColIter colit = spSeq->begcol();
			for(int i=0; i< colneighs; ++i)
			{
				VT * sendbuf = new VT[loclens[i]];
				fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
                
				for(; colit != spSeq->endcol() && colit.colid() < lensums[i+1]; ++colit)	// iterate over a portion of columns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)	// all nonzeros in this column
					{
						sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
					}
				}
                
				VT * recvbuf = NULL;
				if(colrank == i)
				{
					trarr.resize(loclens[i]);
					recvbuf = SpHelper::p2a(trarr);	
				}
				MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), mympiop, i, commGrid->GetColWorld()); // root  = i
				delete [] sendbuf;
			}
			DeleteAll(loclens, lensums);

			GIT reallen;	// Now we have to transpose the vector
			GIT trlen = trarr.size();
			int diagneigh = commGrid->GetComplementRank();
			MPI_Status status;
			MPI_Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ, commGrid->GetWorld(), &status);
	
			rvec.arr.resize(reallen);
			MPI_Sendrecv(SpHelper::p2a(trarr), trlen, MPIType<VT>(), diagneigh, TRX, SpHelper::p2a(rvec.arr), reallen, MPIType<VT>(), diagneigh, TRX, commGrid->GetWorld(), &status);
			rvec.glen = getncol();	// ABAB: Put a sanity check here
			break;

		}
		case Row:	// pack along the rows, result is a vector of size m
		{
			rvec.glen = getnrow();
			int rowneighs = commGrid->GetGridCols();
			int rowrank = commGrid->GetRankInProcRow();
			GIT * loclens = new GIT[rowneighs];
			GIT * lensums = new GIT[rowneighs+1]();	// begin/end points of local lengths
			loclens[rowrank] = rvec.MyLocLength();
			MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetRowWorld());
			partial_sum(loclens, loclens+rowneighs, lensums+1);
			try
			{
				rvec.arr.resize(loclens[rowrank], id);

				// keeping track of all nonzero iterators within columns at once is unscalable w.r.t. memory (due to sqrt(p) scaling)
				// thus we'll do batches of column as opposed to all columns at once. 5 million columns take 80MB (two pointers per column)
				#define MAXCOLUMNBATCH 5 * 1024 * 1024
				typename DER::SpColIter begfinger = spSeq->begcol();	// beginning finger to columns
				
				// Each processor on the same processor row should execute the SAME number of reduce calls
				int numreducecalls = (int) ceil(static_cast<float>(spSeq->getnzc()) / static_cast<float>(MAXCOLUMNBATCH));
				int maxreducecalls;
				MPI_Allreduce( &numreducecalls, &maxreducecalls, 1, MPI_INT, MPI_MAX, commGrid->GetRowWorld());
				
				for(int k=0; k< maxreducecalls; ++k)
				{
					vector<typename DER::SpColIter::NzIter> nziters;
					typename DER::SpColIter curfinger = begfinger; 
					for(; curfinger != spSeq->endcol() && nziters.size() < MAXCOLUMNBATCH ; ++curfinger)	
					{
						nziters.push_back(spSeq->begnz(curfinger));
					}
					for(int i=0; i< rowneighs; ++i)		// step by step to save memory
					{
						VT * sendbuf = new VT[loclens[i]];
						fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
		
						typename DER::SpColIter colit = begfinger;		
						IT colcnt = 0;	// "processed column" counter
						for(; colit != curfinger; ++colit, ++colcnt)	// iterate over this batch of columns until curfinger
						{
							typename DER::SpColIter::NzIter nzit = nziters[colcnt];
							for(; nzit != spSeq->endnz(colit) && nzit.rowid() < lensums[i+1]; ++nzit)	// a portion of nonzeros in this column
							{
								sendbuf[nzit.rowid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[nzit.rowid()-lensums[i]]);
							}
							nziters[colcnt] = nzit;	// set the new finger
						}

						VT * recvbuf = NULL;
						if(rowrank == i)
						{
							for(int j=0; j< loclens[i]; ++j)
							{
								sendbuf[j] = __binary_op(rvec.arr[j], sendbuf[j]);	// rvec.arr will be overriden with MPI_Reduce, save its contents
							}
							recvbuf = SpHelper::p2a(rvec.arr);	
						}
						MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), mympiop, i, commGrid->GetRowWorld()); // root = i
						delete [] sendbuf;
					}
					begfinger = curfinger;	// set the next begfilter
				}
				DeleteAll(loclens, lensums);	
			}
			catch (length_error& le) 
			{
	 			 cerr << "Length error: " << le.what() << endl;
  			}
			break;
		}
		default:
		{
			cout << "Unknown reduction dimension, returning empty vector" << endl;
			break;
		}
	}
}


#define KSELECTLIMIT 50



//! Kselect wrapper for a select columns of the matrix
//! Indices of the input sparse vectors kth denote the queried columns of the matrix
//! Upon return, values of kth stores the kth entries of the queried columns
//! Returns true if Kselect algorithm is invoked for at least one column
//! Otherwise, returns false
template <class IT, class NT, class DER>
template <typename VT, typename GIT>
bool SpParMat<IT,NT,DER>::Kselect(FullyDistSpVec<GIT,VT> & kth, IT k_limit) const
{
    bool ret;
    FullyDistVec<GIT,VT> kthAll ( getcommgrid());
    if(k_limit > KSELECTLIMIT)
    {
        ret = Kselect2(kthAll, k_limit);
    }
    else
    {
        //ret = Kselect1(kthAll, k_limit, myidentity<NT>());
        return Kselect1(kth, k_limit, myidentity<NT>());
    }
    
    //kth.DebugPrint();
    //kthAll.DebugPrint();
    FullyDistSpVec<GIT,VT> temp = EWiseApply<VT>(kth, kthAll,
                                                 [](VT spval, VT dval){return dval;},
                                                 [](VT spval, VT dval){return true;},
                                                 false, NT());
    //temp.DebugPrint();
    kth = temp;
    return ret;
}


//! Returns true if Kselect algorithm is invoked for at least one column
//! Otherwise, returns false
//! if false, rvec contains either the minimum entry in each column or zero
template <class IT, class NT, class DER>
template <typename VT, typename GIT>
bool SpParMat<IT,NT,DER>::Kselect(FullyDistVec<GIT,VT> & rvec, IT k_limit) const
{
/*#ifdef COMBBLAS_DEBUG
    FullyDistVec<GIT,VT> test1(rvec.getcommgrid());
    FullyDistVec<GIT,VT> test2(rvec.getcommgrid());
    Kselect1(test1, k_limit, myidentity<NT>());
    Kselect2(test2, k_limit);
    if(test1 == test2)
        SpParHelper::Print("Kselect1 and Kselect2 producing same results\n");
    else
    {
        SpParHelper::Print("WARNING: Kselect1 and Kselect2 producing DIFFERENT results\n");
        //test1.PrintToFile("test1");
        //test2.PrintToFile("test2");
    }
#endif*/
    
    if(k_limit > KSELECTLIMIT)
    {
        return Kselect2(rvec, k_limit);
    }
    else
    {
        return Kselect1(rvec, k_limit, myidentity<NT>());
    }
}

/* identify the k-th maximum element in each column of a matrix
** if the number of nonzeros in a column is less than k, return minimum entry
** Caution: this is a preliminary implementation: needs 3*(n/sqrt(p))*k memory per processor
** this memory requirement is too high for larger k
 */
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _UnaryOperation>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect1(FullyDistVec<GIT,VT> & rvec, IT k, _UnaryOperation __unary_op) const
{
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    FullyDistVec<IT, IT> nnzPerColumn (getcommgrid());
    Reduce(nnzPerColumn, Column, plus<IT>(), (IT)0, [](NT val){return (IT)1;});
    IT maxnnzPerColumn = nnzPerColumn.Reduce(maximum<IT>(), (IT)0);
    if(k>maxnnzPerColumn)
    {
        SpParHelper::Print("Kselect: k is greater then maxNnzInColumn. Calling Reduce instead...\n");
        Reduce(rvec, Column, minimum<NT>(), static_cast<NT>(0));
        return false;
    }
    
    IT n_thiscol = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)
    
    // check, memory should be min(n_thiscol*k, local nnz)
    // hence we will not overflow for very large k
    vector<VT> sendbuf(n_thiscol*k);
    vector<IT> send_coldisp(n_thiscol+1,0);
    vector<IT> local_coldisp(n_thiscol+1,0);
    
    
    //displacement of local columns
    //local_coldisp is the displacement of all nonzeros per column
    //send_coldisp is the displacement of k nonzeros per column
    IT nzc = 0;
    if(spSeq->getnnz()>0)
    {
        typename DER::SpColIter colit = spSeq->begcol();
        for(IT i=0; i<n_thiscol; ++i)
        {
            local_coldisp[i+1] = local_coldisp[i];
            send_coldisp[i+1] = send_coldisp[i];
            if(i==colit.colid())
            {
                local_coldisp[i+1] += colit.nnz();
                if(colit.nnz()>=k)
                    send_coldisp[i+1] += k;
                else
                    send_coldisp[i+1] += colit.nnz();
                colit++;
                nzc++;
            }
        }
    }
    assert(local_coldisp[n_thiscol] == spSeq->getnnz());
    
    // a copy of local part of the matrix
    // this can be avoided if we write our own local kselect function instead of using partial_sort
    vector<VT> localmat(spSeq->getnnz());


#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i<nzc; i++)
    //for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        typename DER::SpColIter colit = spSeq->begcol() + i;
        IT colid = colit.colid();
        IT idx = local_coldisp[colid];
        for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
        {
            localmat[idx++] = static_cast<VT>(__unary_op(nzit.value()));
        }
        
        if(colit.nnz()<=k)
        {
            sort(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid+1], greater<VT>());
            copy(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid+1], sendbuf.begin()+send_coldisp[colid]);
        }
        else
        {
            partial_sort(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid]+k, localmat.begin()+local_coldisp[colid+1], greater<VT>());
            copy(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid]+k, sendbuf.begin()+send_coldisp[colid]);
        }
    }
    
    vector<VT>().swap(localmat);
    vector<IT>().swap(local_coldisp);

    vector<VT> recvbuf(n_thiscol*k);
    vector<VT> tempbuf(n_thiscol*k);
    vector<IT> recv_coldisp(n_thiscol+1);
    vector<IT> templen(n_thiscol);
    
    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();
    
    for(int p=2; p <= colneighs; p*=2)
    {
       
        if(colrank%p == p/2) // this processor is a sender in this round
        {
            int receiver = colrank - ceil(p/2);
            MPI_Send(send_coldisp.data(), n_thiscol+1, MPIType<IT>(), receiver, 0, commGrid->GetColWorld());
            MPI_Send(sendbuf.data(), send_coldisp[n_thiscol], MPIType<VT>(), receiver, 1, commGrid->GetColWorld());
            //break;
        }
        else if(colrank%p == 0) // this processor is a receiver in this round
        {
            int sender = colrank + ceil(p/2);
            if(sender < colneighs)
            {
                
                MPI_Recv(recv_coldisp.data(), n_thiscol+1, MPIType<IT>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                MPI_Recv(recvbuf.data(), recv_coldisp[n_thiscol], MPIType<VT>(), sender, 1, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                


#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; ++i)
                {
                    // partial merge until first k elements
                    IT j=send_coldisp[i], l=recv_coldisp[i];
                    //IT templen[i] = k*i;
                    IT offset = k*i;
                    IT lid = 0;
                    for(; j<send_coldisp[i+1] && l<recv_coldisp[i+1] && lid<k;)
                    {
                        if(sendbuf[j] > recvbuf[l])  // decision
                            tempbuf[offset+lid++] = sendbuf[j++];
                        else
                            tempbuf[offset+lid++] = recvbuf[l++];
                    }
                    while(j<send_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = sendbuf[j++];
                    while(l<recv_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = recvbuf[l++];
                    templen[i] = lid;
                }
                
                send_coldisp[0] = 0;
                for(IT i=0; i<n_thiscol; i++)
                {
                    send_coldisp[i+1] = send_coldisp[i] + templen[i];
                }
                
               
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; i++) // direct copy
                {
                    IT offset = k*i;
                    copy(tempbuf.begin()+offset, tempbuf.begin()+offset+templen[i], sendbuf.begin() + send_coldisp[i]);
                }
            }
        }
    }
    MPI_Barrier(commGrid->GetWorld());
    vector<VT> kthItem(n_thiscol);

    int root = commGrid->GetDiagOfProcCol();
    if(root==0 && colrank==0) // rank 0
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else if (nitems==0)
                kthItem[i] = numeric_limits<VT>::min(); // return minimum possible value if a column is empty
            else
                kthItem[i] = sendbuf[send_coldisp[i+1]-1]; // returning the last entry if nnz in this column is less than k
        }
    }
    else if(root>0 && colrank==0) // send to the diagonl processor of this processor column
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else if (nitems==0)
                kthItem[i] = numeric_limits<VT>::min(); // return minimum possible value if a column is empty
            else
               kthItem[i] = sendbuf[send_coldisp[i+1]-1]; // returning the last entry if nnz in this column is less than k
        }
        MPI_Send(kthItem.data(), n_thiscol, MPIType<VT>(), root, 0, commGrid->GetColWorld());
    }
    else if(root>0 && colrank==root)
    {
        MPI_Recv(kthItem.data(), n_thiscol, MPIType<VT>(), 0, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
    }
    
    vector <int> sendcnts;
    vector <int> dpls;
    if(colrank==root)
    {
        int proccols = commGrid->GetGridCols();
        IT n_perproc = n_thiscol / proccols;
        sendcnts.resize(proccols);
        fill(sendcnts.data(), sendcnts.data()+proccols-1, n_perproc);
        sendcnts[proccols-1] = n_thiscol - (n_perproc * (proccols-1));
        dpls.resize(proccols,0);	// displacements (zero initialized pid)
        partial_sum(sendcnts.data(), sendcnts.data()+proccols-1, dpls.data()+1);
    }
    
    int rowroot = commGrid->GetDiagOfProcRow();
    int recvcnts = 0;
    // scatter received data size
    MPI_Scatter(sendcnts.data(),1, MPI_INT, & recvcnts, 1, MPI_INT, rowroot, commGrid->GetRowWorld());
    
    rvec.arr.resize(recvcnts);
    MPI_Scatterv(kthItem.data(),sendcnts.data(), dpls.data(), MPIType<VT>(), rvec.arr.data(), rvec.arr.size(), MPIType<VT>(),rowroot, commGrid->GetRowWorld());
    rvec.glen = getncol();
    return true;
}


// TODO: 1. send and receive buffer proportional to active columns
// 2. Check which parts are not maltithreaded. Going from 24 threads/node to 6 t/node make it twice faster
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _UnaryOperation>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect1(FullyDistSpVec<GIT,VT> & rvec, IT k, _UnaryOperation __unary_op) const
{
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    /*
    FullyDistVec<IT, IT> nnzPerColumn (getcommgrid());
    Reduce(nnzPerColumn, Column, plus<IT>(), (IT)0, [](NT val){return (IT)1;});
    IT maxnnzPerColumn = nnzPerColumn.Reduce(maximum<IT>(), (IT)0);
    if(k>maxnnzPerColumn)
    {
        SpParHelper::Print("Kselect: k is greater then maxNnzInColumn. Calling Reduce instead...\n");
        Reduce(rvec, Column, minimum<NT>(), static_cast<NT>(0));
        return false;
    }
     */
    
    IT n_thiscol = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)
    MPI_Comm World = rvec.commGrid->GetWorld();
    MPI_Comm ColWorld = rvec.commGrid->GetColWorld();
    MPI_Comm RowWorld = rvec.commGrid->GetRowWorld();
  
    
    //replicate sparse indices along processor column
    int accnz;
    int32_t trxlocnz;
    GIT lenuntil;
    int32_t *trxinds, *indacc;
    VT *trxnums, *numacc;
    TransposeVector(World, rvec, trxlocnz, lenuntil, trxinds, trxnums, true);
    
    if(rvec.commGrid->GetGridRows() > 1)
    {
        //TODO: we only need to communicate indices
        AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, true);  // trxindS/trxnums deallocated, indacc/numacc allocated, accnz set
    }
    else
    {
        accnz = trxlocnz;
        indacc = trxinds;     //aliasing ptr
        numacc = trxnums;     //aliasing ptr
    }
    
    
    vector<bool> isactive(n_thiscol,false);
    for(int i=0; i<accnz ; i++)
    {
        isactive[indacc[i]] = true;
        //cout << indacc[i] <<  " ";
    }
    IT nActiveCols = accnz;//count_if(isactive.begin(), isactive.end(), [](bool ac){return ac;});
    // check, memory should be min(n_thiscol*k, local nnz)
    // hence we will not overflow for very large k
    
    vector<IT> send_coldisp(n_thiscol+1,0);
    vector<IT> local_coldisp(n_thiscol+1,0);
    //vector<VT> sendbuf(nActiveCols*k);
    VT * sendbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));
    
    
    //displacement of local columns
    //local_coldisp is the displacement of all nonzeros per column
    //send_coldisp is the displacement of k nonzeros per column
    IT nzc = 0;
    if(spSeq->getnnz()>0)
    {
        typename DER::SpColIter colit = spSeq->begcol();
        for(IT i=0; i<n_thiscol; ++i)
        {
            local_coldisp[i+1] = local_coldisp[i];
            send_coldisp[i+1] = send_coldisp[i];
            if(i==colit.colid())
            {
                if(isactive[i])
                {
                    local_coldisp[i+1] += colit.nnz();
                    if(colit.nnz()>=k)
                        send_coldisp[i+1] += k;
                    else
                        send_coldisp[i+1] += colit.nnz();
                }
                colit++;
                nzc++;
            }
        }
    }
    
    // a copy of local part of the matrix
    // this can be avoided if we write our own local kselect function instead of using partial_sort
    //vector<VT> localmat(local_coldisp[n_thiscol]);
    VT * localmat = static_cast<VT *> (::operator new (local_coldisp[n_thiscol]*sizeof(VT)));
    
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i<nzc; i++)
        //for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        typename DER::SpColIter colit = spSeq->begcol() + i;
        IT colid = colit.colid();
        if(isactive[colid])
        {
            IT idx = local_coldisp[colid];
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
            {
                localmat[idx++] = static_cast<VT>(__unary_op(nzit.value()));
            }
            
            if(colit.nnz()<=k)
            {
                sort(localmat+local_coldisp[colid], localmat+local_coldisp[colid+1], greater<VT>());
                copy(localmat+local_coldisp[colid], localmat+local_coldisp[colid+1], sendbuf+send_coldisp[colid]);
            }
            else
            {
                partial_sort(localmat+local_coldisp[colid], localmat+local_coldisp[colid]+k, localmat+local_coldisp[colid+1], greater<VT>());
                copy(localmat+local_coldisp[colid], localmat+local_coldisp[colid]+k, sendbuf+send_coldisp[colid]);
            }
        }
    }

    
    //vector<VT>().swap(localmat);
    ::operator delete(localmat);
    vector<IT>().swap(local_coldisp);
    
    VT * recvbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));
    VT * tempbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));
    //vector<VT> recvbuf(n_thiscol*k);
    //vector<VT> tempbuf(n_thiscol*k);
    vector<IT> recv_coldisp(n_thiscol+1);
    vector<IT> templen(n_thiscol);
    
    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();
    
    for(int p=2; p <= colneighs; p*=2)
    {
        
        if(colrank%p == p/2) // this processor is a sender in this round
        {
            int receiver = colrank - ceil(p/2);
            MPI_Send(send_coldisp.data(), n_thiscol+1, MPIType<IT>(), receiver, 0, commGrid->GetColWorld());
            MPI_Send(sendbuf, send_coldisp[n_thiscol], MPIType<VT>(), receiver, 1, commGrid->GetColWorld());
            //break;
        }
        else if(colrank%p == 0) // this processor is a receiver in this round
        {
            int sender = colrank + ceil(p/2);
            if(sender < colneighs)
            {
                
                MPI_Recv(recv_coldisp.data(), n_thiscol+1, MPIType<IT>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                MPI_Recv(recvbuf, recv_coldisp[n_thiscol], MPIType<VT>(), sender, 1, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                
                
                
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; ++i)
                {
                    // partial merge until first k elements
                    IT j=send_coldisp[i], l=recv_coldisp[i];
                    //IT templen[i] = k*i;
                    IT offset = k*i;
                    IT lid = 0;
                    for(; j<send_coldisp[i+1] && l<recv_coldisp[i+1] && lid<k;)
                    {
                        if(sendbuf[j] > recvbuf[l])  // decision
                            tempbuf[offset+lid++] = sendbuf[j++];
                        else
                            tempbuf[offset+lid++] = recvbuf[l++];
                    }
                    while(j<send_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = sendbuf[j++];
                    while(l<recv_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = recvbuf[l++];
                    templen[i] = lid;
                }
                
                send_coldisp[0] = 0;
                for(IT i=0; i<n_thiscol; i++)
                {
                    send_coldisp[i+1] = send_coldisp[i] + templen[i];
                }
                
                
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; i++) // direct copy
                {
                    IT offset = k*i;
                    copy(tempbuf+offset, tempbuf+offset+templen[i], sendbuf + send_coldisp[i]);
                }
            }
        }
    }
    MPI_Barrier(commGrid->GetWorld());
    vector<VT> kthItem(n_thiscol);
    
    int root = commGrid->GetDiagOfProcCol();
    if(root==0 && colrank==0) // rank 0
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else if (nitems==0)
                kthItem[i] = numeric_limits<VT>::min(); // return minimum possible value if a column is empty
            else
                kthItem[i] = sendbuf[send_coldisp[i+1]-1]; // returning the last entry if nnz in this column is less than k
        }
    }
    else if(root>0 && colrank==0) // send to the diagonl processor of this processor column
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else if (nitems==0)
                kthItem[i] = numeric_limits<VT>::min(); // return minimum possible value if a column is empty
            else
                kthItem[i] = sendbuf[send_coldisp[i+1]-1]; // returning the last entry if nnz in this column is less than k
        }
        MPI_Send(kthItem.data(), n_thiscol, MPIType<VT>(), root, 0, commGrid->GetColWorld());
    }
    else if(root>0 && colrank==root)
    {
        MPI_Recv(kthItem.data(), n_thiscol, MPIType<VT>(), 0, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
    }
    
    ::operator delete(sendbuf);
    ::operator delete(recvbuf);
    ::operator delete(tempbuf);
    
    vector <int> sendcnts;
    vector <int> dpls;
    vector<VT> kthItemActive(nActiveCols);

    if(colrank==root)
    {
        int proccols = commGrid->GetGridCols();
        IT n_perproc = n_thiscol / proccols;
        
        sendcnts.resize(proccols);
        fill(sendcnts.data(), sendcnts.data()+proccols-1, n_perproc);
        sendcnts[proccols-1] = n_thiscol - (n_perproc * (proccols-1));
        dpls.resize(proccols,0);	// displacements (zero initialized pid)
        partial_sum(sendcnts.data(), sendcnts.data()+proccols-1, dpls.data()+1);
    }
    
    int rowroot = commGrid->GetDiagOfProcRow();
    int recvcnts = 0;
    // scatter received data size
    
    MPI_Scatter(sendcnts.data(),1, MPI_INT, & recvcnts, 1, MPI_INT, rowroot, commGrid->GetRowWorld());

    
    
    // first populate the dense vector
    FullyDistVec<GIT,VT> dvec(rvec.getcommgrid());
    dvec.arr.resize(recvcnts);
    MPI_Scatterv(kthItem.data(),sendcnts.data(), dpls.data(), MPIType<VT>(), dvec.arr.data(), dvec.arr.size(), MPIType<VT>(),rowroot, commGrid->GetRowWorld());
    dvec.glen = getncol();
    
    //populate the sparse vector
    rvec = EWiseApply<VT>(rvec, dvec,
                               [](VT sv, VT dv){return dv;},
                               [](VT sv, VT dv){return true;},
                               false, static_cast<VT>(0));
     
    return true;
}

// only defined for symmetric matrix
template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::Bandwidth() const
{
    IT upperlBW = -1;
    IT lowerlBW = -1;
    IT m_perproc = getnrow() / commGrid->GetGridRows();
    IT n_perproc = getncol() / commGrid->GetGridCols();
    IT moffset = commGrid->GetRankInProcCol() * m_perproc;
    IT noffset = commGrid->GetRankInProcRow() * n_perproc;
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        IT diagrow = colit.colid() + noffset;
        typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
        if(nzit != spSeq->endnz(colit)) // nonempty column
        {
            IT firstrow = nzit.rowid() + moffset;
            IT lastrow = (nzit+ colit.nnz()-1).rowid() + moffset;
           
            if(firstrow <= diagrow) // upper diagonal
            {
                IT dev = diagrow - firstrow;
                if(upperlBW < dev) upperlBW = dev;
            }
            if(lastrow >= diagrow) // lower diagonal
            {
                IT dev = lastrow - diagrow;
                if(lowerlBW < dev) lowerlBW = dev;
            }
        }
    }
    IT upperBW;
    //IT lowerBW;
    MPI_Allreduce( &upperlBW, &upperBW, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    //MPI_Allreduce( &lowerlBW, &lowerBW, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    
    //return (upperBW + lowerBW + 1);
    return (upperBW);
}



// only defined for symmetric matrix
template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::Profile() const
{
    int colrank = commGrid->GetRankInProcRow();
    IT cols = getncol();
    IT rows = getnrow();
    IT m_perproc = cols / commGrid->GetGridRows();
    IT n_perproc = rows / commGrid->GetGridCols();
    IT moffset = commGrid->GetRankInProcCol() * m_perproc;
    IT noffset = colrank * n_perproc;
  

    int pc = commGrid->GetGridCols();
    IT n_thisproc;
    if(colrank!=pc-1 ) n_thisproc = n_perproc;
    else n_thisproc =  cols - (pc-1)*n_perproc;
 
    
    vector<IT> firstRowInCol(n_thisproc,getnrow());
    vector<IT> lastRowInCol(n_thisproc,-1);
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        IT diagrow = colit.colid() + noffset;
        typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
        if(nzit != spSeq->endnz(colit)) // nonempty column
        {
            IT firstrow = nzit.rowid() + moffset;
            IT lastrow = (nzit+ colit.nnz()-1).rowid() + moffset;
            if(firstrow <= diagrow) // upper diagonal
            {
                firstRowInCol[colit.colid()] = firstrow;
            }
            if(lastrow >= diagrow) // lower diagonal
            {
                lastRowInCol[colit.colid()] = lastrow;
            }
        }
    }
    
    vector<IT> firstRowInCol_global(n_thisproc,getnrow());
    //vector<IT> lastRowInCol_global(n_thisproc,-1);
    MPI_Allreduce( firstRowInCol.data(), firstRowInCol_global.data(), n_thisproc, MPIType<IT>(), MPI_MIN, commGrid->colWorld);
    //MPI_Allreduce( lastRowInCol.data(), lastRowInCol_global.data(), n_thisproc, MPIType<IT>(), MPI_MAX, commGrid->GetColWorld());
    
    IT profile = 0;
    for(IT i=0; i<n_thisproc; i++)
    {
        if(firstRowInCol_global[i]==rows) // empty column
            profile++;
        else
            profile += (i + noffset - firstRowInCol_global[i]);
    }
    
    IT profile_global = 0;
    MPI_Allreduce( &profile, &profile_global, 1, MPIType<IT>(), MPI_SUM, commGrid->rowWorld);
    
    return (profile_global);
}



template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation>
void SpParMat<IT,NT,DER>::MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, bool exclude) const
{
    if (dim!=Column)
    {
        SpParHelper::Print("SpParMat::MaskedReduce() is only implemented for Colum\n");
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    MaskedReduce(rvec, mask, dim, __binary_op, id, myidentity<NT>(), exclude);
}

/**
 * Reduce along the column into a vector
 * @param[in] mask {A sparse vector indicating row indices included/excluded (based on exclude argument) in the reduction }
 * @param[in] __binary_op {the operation used for reduction; examples: max, min, plus, multiply, and, or. Its parameters and return type are all VT}
 * @param[in] id {scalar that is used as the identity for __binary_op; examples: zero, infinity}
 * @param[in] __unary_op {optional unary operation applied to nonzeros *before* the __binary_op; examples: 1/x, x^2}
 * @param[in] exclude {if true, masked row indices are included in the reduction}
 * @param[out] rvec {the return vector, specified as an output parameter to allow arbitrary return types via VT}
 **/
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, bool exclude) const
{
    MPI_Comm World = commGrid->GetWorld();
    MPI_Comm ColWorld = commGrid->GetColWorld();
    MPI_Comm RowWorld = commGrid->GetRowWorld();

    if (dim!=Column)
    {
        SpParHelper::Print("SpParMat::MaskedReduce() is only implemented for Colum\n");
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::MaskedReduce() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    int rowneighs = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();
    vector<int> rownz(rowneighs);
    int locnnzMask = static_cast<int> (mask.getlocnnz());
    rownz[rowrank] = locnnzMask;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rownz.data(), 1, MPI_INT, RowWorld);
    vector<int> dpls(rowneighs+1,0);
    std::partial_sum(rownz.begin(), rownz.end(), dpls.data()+1);
    int accnz = std::accumulate(rownz.begin(), rownz.end(), 0);
    vector<GIT> sendInd(locnnzMask);
    transform(mask.ind.begin(), mask.ind.end(),sendInd.begin(), bind2nd(plus<GIT>(), mask.RowLenUntil()));
    
    vector<GIT> indMask(accnz);
    MPI_Allgatherv(sendInd.data(), rownz[rowrank], MPIType<GIT>(), indMask.data(), rownz.data(), dpls.data(), MPIType<GIT>(), RowWorld);
    
    
    // We can't use rvec's distribution (rows first, columns later) here
    IT n_thiscol = getlocalcols();   // length assigned to this processor column
    int colneighs = commGrid->GetGridRows();	// including oneself
    int colrank = commGrid->GetRankInProcCol();
    
    GIT * loclens = new GIT[colneighs];
    GIT * lensums = new GIT[colneighs+1]();	// begin/end points of local lengths
    
    GIT n_perproc = n_thiscol / colneighs;    // length on a typical processor
    if(colrank == colneighs-1)
        loclens[colrank] = n_thiscol - (n_perproc*colrank);
    else
        loclens[colrank] = n_perproc;
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetColWorld());
    partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits
    
    vector<VT> trarr;
    typename DER::SpColIter colit = spSeq->begcol();
    for(int i=0; i< colneighs; ++i)
    {
        VT * sendbuf = new VT[loclens[i]];
        fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
        
        for(; colit != spSeq->endcol() && colit.colid() < lensums[i+1]; ++colit)	// iterate over a portion of columns
        {
            int k=0;
            typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
            
            for(; nzit != spSeq->endnz(colit) && k < indMask.size(); )	// all nonzeros in this column
            {
                if(nzit.rowid() < indMask[k])
                {
                    if(exclude)
                    {
                        sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    }
                    ++nzit;
                }
                else if(nzit.rowid() > indMask[k]) ++k;
                else
                {
                    if(!exclude)
                    {
                        sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    }
                    ++k;
                    ++nzit;
                }
                
            }
            if(exclude)
            {
                while(nzit != spSeq->endnz(colit))
                {
                    sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    ++nzit;
                }
            }
        }
        
        VT * recvbuf = NULL;
        if(colrank == i)
        {
            trarr.resize(loclens[i]);
            recvbuf = SpHelper::p2a(trarr);
        }
        MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), i, commGrid->GetColWorld()); // root  = i
        delete [] sendbuf;
    }
    DeleteAll(loclens, lensums);
    
    GIT reallen;	// Now we have to transpose the vector
    GIT trlen = trarr.size();
    int diagneigh = commGrid->GetComplementRank();
    MPI_Status status;
    MPI_Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ, commGrid->GetWorld(), &status);
    
    rvec.arr.resize(reallen);
    MPI_Sendrecv(SpHelper::p2a(trarr), trlen, MPIType<VT>(), diagneigh, TRX, SpHelper::p2a(rvec.arr), reallen, MPIType<VT>(), diagneigh, TRX, commGrid->GetWorld(), &status);
    rvec.glen = getncol();	// ABAB: Put a sanity check here
    
}




template <class IT, class NT, class DER>
template <typename NNT,typename NDER>
SpParMat<IT,NT,DER>::operator SpParMat<IT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<IT,NNT,NDER> (convert, commGrid);
}

//! Change index type as well
template <class IT, class NT, class DER>
template <typename NIT, typename NNT,typename NDER>
SpParMat<IT,NT,DER>::operator SpParMat<NIT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<NIT,NNT,NDER> (convert, commGrid);
}

/**
 * Create a submatrix of size m x (size(ci) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 */
template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRefCol (const vector<IT> & ci) const
{
	vector<IT> ri;
	DER * tempseq = new DER((*spSeq)(ri, ci)); 
	return SpParMat<IT,NT,DER> (tempseq, commGrid);	
} 

/** 
 * Generalized sparse matrix indexing (ri/ci are 0-based indexed)
 * Both the storage and the actual values in FullyDistVec should be IT
 * The index vectors are dense and FULLY distributed on all processors
 * We can use this function to apply a permutation like A(p,q) 
 * Sequential indexing subroutine (via multiplication) is general enough.
 */
template <class IT, class NT, class DER>
template <typename PTNTBOOL, typename PTBOOLNT>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRef_SR (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace)
{
	// infer the concrete type SpMat<IT,IT>
	typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;

	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, SpRef fails !"); 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *max_element(ci.arr.begin(), ci.arr.end());

	IT total_m = getnrow();
	IT total_n = getncol();
	if(locmax_ri > total_m || locmax_ci > total_n)	
	{
		throw outofrangeexception();
	}

	// The indices for FullyDistVec are offset'd to 1/p pieces
	// The matrix indices are offset'd to 1/sqrt(p) pieces
	// Add the corresponding offset before sending the data 
	IT roffset = ri.RowLenUntil();
	IT rrowlen = ri.MyRowLength();
	IT coffset = ci.RowLenUntil();
	IT crowlen = ci.MyRowLength();

	// We create two boolean matrices P and Q
	// Dimensions:  P is size(ri) x m
	//		Q is n x size(ci) 
	// Range(ri) = {0,...,m-1}
	// Range(ci) = {0,...,n-1}

	IT rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
	IT totalm = getnrow();	// collective call
	IT totaln = getncol();
	IT m_perproccol = totalm / rowneighs;
	IT n_perproccol = totaln / rowneighs;

	// Get the right local dimensions
	IT diagneigh = commGrid->GetComplementRank();
	IT mylocalrows = getlocalrows();
	IT mylocalcols = getlocalcols();
	IT trlocalrows;
	MPI_Status status;
	MPI_Sendrecv(&mylocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, &trlocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, commGrid->GetWorld(), &status);
	// we don't need trlocalcols because Q.Transpose() will take care of it

	vector< vector<IT> > rowid(rowneighs);	// reuse for P and Q 
	vector< vector<IT> > colid(rowneighs);

	// Step 1: Create P
	IT locvec = ri.arr.size();	// nnz in local vector
	for(typename vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		// recipient alone progessor row
		IT rowrec = (m_perproccol!=0) ? std::min(ri.arr[i] / m_perproccol, rowneighs-1) : (rowneighs-1);	

		// ri's numerical values give the colids and its local indices give rowids
		rowid[rowrec].push_back( i + roffset);	
		colid[rowrec].push_back(ri.arr[i] - (rowrec * m_perproccol));
	}

	int * sendcnt = new int[rowneighs];	// reuse in Q as well
	int * recvcnt = new int[rowneighs];
	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetRowWorld()); // share the counts
	int * sdispls = new int[rowneighs]();
	int * rdispls = new int[rowneighs]();
	partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT p_nnz = accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * p_rows = new IT[p_nnz];
	IT * p_cols = new IT[p_nnz];
  	IT * senddata = new IT[locvec];	// re-used for both rows and columns
	for(int i=0; i<rowneighs; ++i)
	{
		copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());

	for(int i=0; i<rowneighs; ++i)
	{
		copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
	delete [] senddata;

	tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz]; 
	for(IT i=0; i< p_nnz; ++i)
	{
		p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
	}
	DeleteAll(p_rows, p_cols);

	DER_IT * PSeq = new DER_IT(); 
	PSeq->Create( p_nnz, rrowlen, trlocalrows, p_tuples);		// deletion of tuples[] is handled by SpMat::Create

	SpParMat<IT,NT,DER> PA(commGrid);
	if(&ri == &ci)	// Symmetric permutation
	{
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
		#ifdef SPREFDEBUG
		SpParHelper::Print("Symmetric permutation\n", commGrid->GetWorld());
		#endif
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);
		if(inplace) 
		{
			#ifdef SPREFDEBUG	
			SpParHelper::Print("In place multiplication\n", commGrid->GetWorld());
			#endif
        		*this = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P, *this, false, true);	// clear the memory of *this

			//ostringstream outb;
			//outb << "P_after_" << commGrid->myrank;
			//ofstream ofb(outb.str().c_str());
			//P.put(ofb);

			P.Transpose();	
       	 		*this = Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(*this, P, true, true);	// clear the memory of both *this and P
			return SpParMat<IT,NT,DER>(commGrid);	// dummy return to match signature
		}
		else
		{
			PA = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P,*this);
			P.Transpose();
			return Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, P);
		}
	}
	else
	{
		// Intermediate step (to save memory): Form PA and store it in P
		// Distributed matrix generation (collective call)
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);

		// Do parallel matrix-matrix multiply
        	PA = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P, *this);
	}	// P is destructed here
#ifndef NDEBUG
	PA.PrintInfo();
#endif
	// Step 2: Create Q  (use the same row-wise communication and transpose at the end)
	// This temporary to-be-transposed Q is size(ci) x n 
	locvec = ci.arr.size();	// nnz in local vector (reset variable)
	for(typename vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		IT rowrec = (n_perproccol!=0) ? std::min(ci.arr[i] / n_perproccol, rowneighs-1) : (rowneighs-1);	

		// ri's numerical values give the colids and its local indices give rowids
		rowid[rowrec].push_back( i + coffset);	
		colid[rowrec].push_back(ci.arr[i] - (rowrec * n_perproccol));
	}

	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();	// update with new sizes

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetRowWorld()); // share the counts
	fill(sdispls, sdispls+rowneighs, 0);	// reset
	fill(rdispls, rdispls+rowneighs, 0);
	partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT q_nnz = accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * q_rows = new IT[q_nnz];
	IT * q_cols = new IT[q_nnz];
  	senddata = new IT[locvec];	
	for(int i=0; i<rowneighs; ++i)
	{
		copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_rows, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());

	for(int i=0; i<rowneighs; ++i)
	{
		copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_cols, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);

	tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
	for(IT i=0; i< q_nnz; ++i)
	{
		q_tuples[i] = make_tuple(q_rows[i], q_cols[i], 1);
	}
	DeleteAll(q_rows, q_cols);
	DER_IT * QSeq = new DER_IT(); 
	QSeq->Create( q_nnz, crowlen, mylocalcols, q_tuples);		// Creating Q' instead

	// Step 3: Form PAQ
	// Distributed matrix generation (collective call)
	SpParMat<IT,bool,DER_IT> Q (QSeq, commGrid);
	Q.Transpose();	
	if(inplace)
	{
       		*this = Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, Q, true, true);	// clear the memory of both PA and P
		return SpParMat<IT,NT,DER>(commGrid);	// dummy return to match signature
	}
	else
	{
        	return Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, Q);
	}
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::SpAsgn(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, SpParMat<IT,NT,DER> & B)
{
	typedef PlusTimesSRing<NT, NT> PTRing;
	
	if((*(ri.commGrid) != *(B.commGrid)) || (*(ci.commGrid) != *(B.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT total_m_A = getnrow();
	IT total_n_A = getncol();
	IT total_m_B = B.getnrow();
	IT total_n_B = B.getncol();
	
	if(total_m_B != ri.TotalLength())
	{
		SpParHelper::Print("First dimension of B does NOT match the length of ri, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	if(total_n_B != ci.TotalLength())
	{
		SpParHelper::Print("Second dimension of B does NOT match the length of ci, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	Prune(ri, ci);	// make a hole	
	
	// embed B to the size of A
	FullyDistVec<IT,IT> * rvec = new FullyDistVec<IT,IT>(ri.commGrid);
	rvec->iota(total_m_B, 0);	// sparse() expects a zero based index
	
	SpParMat<IT,NT,DER> R(total_m_A, total_m_B, ri, *rvec, 1);
	delete rvec;	// free memory
	SpParMat<IT,NT,DER> RB = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(R, B, true, false); // clear memory of R but not B
	
	FullyDistVec<IT,IT> * qvec = new FullyDistVec<IT,IT>(ri.commGrid);
	qvec->iota(total_n_B, 0);
	SpParMat<IT,NT,DER> Q(total_n_B, total_n_A, *qvec, ci, 1);
	delete qvec;	// free memory
	SpParMat<IT,NT,DER> RBQ = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(RB, Q, true, true); // clear memory of RB and Q
	*this += RBQ;	// extend-add
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Prune(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci)
{
	typedef PlusTimesSRing<NT, NT> PTRing;

	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Prune fails!\n", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *max_element(ci.arr.begin(), ci.arr.end());

	IT total_m = getnrow();
	IT total_n = getncol();
	if(locmax_ri > total_m || locmax_ci > total_n)	
	{
		throw outofrangeexception();
	}

	SpParMat<IT,NT,DER> S(total_m, total_m, ri, ri, 1);
	SpParMat<IT,NT,DER> SA = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(S, *this, true, false); // clear memory of S but not *this

	SpParMat<IT,NT,DER> T(total_n, total_n, ci, ci, 1);
	SpParMat<IT,NT,DER> SAT = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(SA, T, true, true); // clear memory of SA and T
	EWiseMult(SAT, true);	// In-place EWiseMult with not(SAT)
}

//! Prune every column of a sparse matrix based on pvals
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::PruneColumn(const FullyDistVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if(getncol() != pvals.TotalLength())
    {
        ostringstream outs;
        outs << "Can not prune column-by-column, dimensions does not match"<< endl;
        outs << getncol() << " != " << pvals.TotalLength() << endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    if(! ( *(getcommgrid()) == *(pvals.getcommgrid())) )
    {
        cout << "Grids are not comparable for PurneColumn" << endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    MPI_Comm World = pvals.commGrid->GetWorld();
    MPI_Comm ColWorld = pvals.commGrid->GetColWorld();
    
    int xsize = (int) pvals.LocArrSize();
    int trxsize = 0;

    
    int diagneigh = pvals.commGrid->GetComplementRank();
    MPI_Status status;
    MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);


    NT * trxnums = new NT[trxsize];
    MPI_Sendrecv(const_cast<NT*>(SpHelper::p2a(pvals.arr)), xsize, MPIType<NT>(), diagneigh, TRX, trxnums, trxsize, MPIType<NT>(), diagneigh, TRX, World, &status);
    
    int colneighs, colrank;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_rank(ColWorld, &colrank);
    int * colsize = new int[colneighs];
    colsize[colrank] = trxsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, ColWorld);
    int * dpls = new int[colneighs]();	// displacements (zero initialized pid)
    std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
    int accsize = std::accumulate(colsize, colsize+colneighs, 0);
    vector<NT> numacc(accsize);

#ifdef COMBBLAS_DEBUG
    ostringstream outs2; 
    outs2 << "PruneColumn displacements: ";
    for(int i=0; i< colneighs; ++i)
    {
	outs2 << dpls[i] << " ";
    }
    outs2 << endl;
    SpParHelper::Print(outs2.str());
    MPI_Barrier(World);
#endif
    
    
    MPI_Allgatherv(trxnums, trxsize, MPIType<NT>(), numacc.data(), colsize, dpls, MPIType<NT>(), ColWorld);
    delete [] trxnums;
    delete [] colsize;
    delete [] dpls;

    //sanity check
    assert(accsize == getlocalcols());
    if (inPlace)
    {
        spSeq->PruneColumn(numacc.data(), __binary_op, inPlace);
        return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
    }
    else
    {
        return SpParMat<IT,NT,DER>(spSeq->PruneColumn(numacc.data(), __binary_op, inPlace), commGrid);
    }
}


//! Prune columns of a sparse matrix selected by nonzero indices of pvals
//! Each selected column is pruned by corresponding values in pvals
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::PruneColumn(const FullyDistSpVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if(getncol() != pvals.TotalLength())
    {
        ostringstream outs;
        outs << "Can not prune column-by-column, dimensions does not match"<< endl;
        outs << getncol() << " != " << pvals.TotalLength() << endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    if(! ( *(getcommgrid()) == *(pvals.getcommgrid())) )
    {
        cout << "Grids are not comparable for PurneColumn" << endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    MPI_Comm World = pvals.commGrid->GetWorld();
    MPI_Comm ColWorld = pvals.commGrid->GetColWorld();
    int diagneigh = pvals.commGrid->GetComplementRank();
    
    IT xlocnz = pvals.getlocnnz();
    IT roffst = pvals.RowLenUntil();
    IT roffset;
    IT trxlocnz = 0;
    
    MPI_Status status;
    MPI_Sendrecv(&roffst, 1, MPIType<IT>(), diagneigh, TROST, &roffset, 1, MPIType<IT>(), diagneigh, TROST, World, &status);
    MPI_Sendrecv(&xlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, World, &status);
    
    vector<IT> trxinds (trxlocnz);
    vector<NT> trxnums (trxlocnz);
    MPI_Sendrecv(pvals.ind.data(), xlocnz, MPIType<IT>(), diagneigh, TRI, trxinds.data(), trxlocnz, MPIType<IT>(), diagneigh, TRI, World, &status);
    MPI_Sendrecv(pvals.num.data(), xlocnz, MPIType<NT>(), diagneigh, TRX, trxnums.data(), trxlocnz, MPIType<NT>(), diagneigh, TRX, World, &status);
    transform(trxinds.data(), trxinds.data()+trxlocnz, trxinds.data(), bind2nd(plus<IT>(), roffset));

    
    int colneighs, colrank;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_rank(ColWorld, &colrank);
    int * colnz = new int[colneighs];
    colnz[colrank] = trxlocnz;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colnz, 1, MPI_INT, ColWorld);
    int * dpls = new int[colneighs]();	// displacements (zero initialized pid)
    std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
    IT accnz = std::accumulate(colnz, colnz+colneighs, 0);
 
    vector<IT> indacc(accnz);
    vector<NT> numacc(accnz);
    MPI_Allgatherv(trxinds.data(), trxlocnz, MPIType<IT>(), indacc.data(), colnz, dpls, MPIType<IT>(), ColWorld);
    MPI_Allgatherv(trxnums.data(), trxlocnz, MPIType<NT>(), numacc.data(), colnz, dpls, MPIType<NT>(), ColWorld);
    
    delete [] colnz;
    delete [] dpls;
    

    if (inPlace)
    {
        spSeq->PruneColumn(indacc.data(), numacc.data(), __binary_op, inPlace);
        return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
    }
    else
    {
        return SpParMat<IT,NT,DER>(spSeq->PruneColumn(indacc.data(), numacc.data(), __binary_op, inPlace), commGrid);
    }
}



// In-place version where rhs type is the same (no need for type promotion)
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::EWiseMult (const SpParMat< IT,NT,DER >  & rhs, bool exclude)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->EWiseMult(*(rhs.spSeq), exclude);		// Dimension compatibility check performed by sequential function
	}
	else
	{
		cout << "Grids are not comparable, EWiseMult() fails !" << endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}	
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::EWiseScale(const DenseParMat<IT, NT> & rhs)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->EWiseScale(rhs.array, rhs.m, rhs.n);	// Dimension compatibility check performed by sequential function
	}
	else
	{
		cout << "Grids are not comparable, EWiseScale() fails !" << endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>
void SpParMat<IT,NT,DER>::UpdateDense(DenseParMat<IT, NT> & rhs, _BinaryOperation __binary_op) const
{
	if(*commGrid == *rhs.commGrid)	
	{
		if(getlocalrows() == rhs.m  && getlocalcols() == rhs.n)
		{
			spSeq->UpdateDense(rhs.array, __binary_op);
		}
		else
		{
			cout << "Matrices have different dimensions, UpdateDense() fails !" << endl;
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
	}
	else
	{
		cout << "Grids are not comparable, UpdateDense() fails !" << endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::PrintInfo() const
{
	IT mm = getnrow(); 
	IT nn = getncol();
	IT nznz = getnnz();
	
	if (commGrid->myrank == 0)	
		cout << "As a whole: " << mm << " rows and "<< nn <<" columns and "<<  nznz << " nonzeros" << endl;
    
#ifdef DEBUG
	IT allprocs = commGrid->grrows * commGrid->grcols;
	for(IT i=0; i< allprocs; ++i)
	{
		if (commGrid->myrank == i)
		{
			cout << "Processor (" << commGrid->GetRankInProcRow() << "," << commGrid->GetRankInProcCol() << ")'s data: " << endl;
			spSeq->PrintInfo();
		}
		MPI_Barrier(commGrid->GetWorld());
	}
#endif
}

template <class IT, class NT, class DER>
bool SpParMat<IT,NT,DER>::operator== (const SpParMat<IT,NT,DER> & rhs) const
{
	int local = static_cast<int>((*spSeq) == (*(rhs.spSeq)));
	int whole = 1;
	MPI_Allreduce( &local, &whole, 1, MPI_INT, MPI_BAND, commGrid->GetWorld());
	return static_cast<bool>(whole);	
}


/**
 ** Private function that carries code common to different sparse() constructors
 ** Before this call, commGrid is already set
 **/
template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename LIT>
void SpParMat< IT,NT,DER >::SparseCommon(vector< vector < tuple<LIT,LIT,NT> > > & data, LIT locsize, IT total_m, IT total_n, _BinaryOperation BinOp)
{
	int nprocs = commGrid->GetSize();
	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data[i].size();	// sizes are all the same

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	

#ifdef COMBBLAS_DEBUG
	IT * gsizes;
	if(commGrid->GetRank() == 0) gsizes = new IT[nprocs];
    	MPI_Gather(&totrecv, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0, commGrid->GetWorld());
	if(commGrid->GetRank() == 0) { copy(gsizes, gsizes+nprocs, ostream_iterator<IT>(cout, " "));   cout << endl; }
	MPI_Barrier(commGrid->GetWorld());
#endif

  	tuple<LIT,LIT,NT> * senddata = new tuple<LIT,LIT,NT>[locsize];	// re-used for both rows and columns
	for(int i=0; i<nprocs; ++i)
	{
		copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
		vector< tuple<LIT,LIT,NT> >().swap(data[i]);	// clear memory
	}
	MPI_Datatype MPI_triple;
	MPI_Type_contiguous(sizeof(tuple<LIT,LIT,NT>), MPI_CHAR, &MPI_triple);
	MPI_Type_commit(&MPI_triple);

	tuple<LIT,LIT,NT> * recvdata = new tuple<LIT,LIT,NT>[totrecv];	
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPI_triple, recvdata, recvcnt, rdispls, MPI_triple, commGrid->GetWorld());

	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);
	MPI_Type_free(&MPI_triple);

	int r = commGrid->GetGridRows();
	int s = commGrid->GetGridCols();
	IT m_perproc = total_m / r;
	IT n_perproc = total_n / s;
	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	IT locrows, loccols; 
	if(myprocrow != r-1)	locrows = m_perproc;
	else 	locrows = total_m - myprocrow * m_perproc;
	if(myproccol != s-1)	loccols = n_perproc;
	else	loccols = total_n - myproccol * n_perproc;
    
	SpTuples<LIT,NT> A(totrecv, locrows, loccols, recvdata);	// It is ~SpTuples's job to deallocate
	
    	// the previous constructor sorts based on columns-first (but that doesn't matter as long as they are sorted one way or another)
    	A.RemoveDuplicates(BinOp);
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}


//! All vectors are zero-based indexed (as usual)
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const FullyDistVec<IT,NT> & distvals, bool SumDuplicates)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails!\n");  // commGrid is not initialized yet
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) || (distcols.TotalLength() != distvals.TotalLength()))
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails!");
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}

	commGrid = distrows.commGrid;	
	int nprocs = commGrid->GetSize();
	vector< vector < tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(make_tuple(lrow,lcol,distvals.arr[i]));	
	}
    if(SumDuplicates)
    {
        SparseCommon(data, locsize, total_m, total_n, plus<NT>());
    }
    else
    {
        SparseCommon(data, locsize, total_m, total_n, maximum<NT>());
    }
}



template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const NT & val, bool SumDuplicates)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) )
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails!\n");
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) )
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails!\n");
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	commGrid = distrows.commGrid;
	int nprocs = commGrid->GetSize();
	vector< vector < tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(make_tuple(lrow,lcol,val));	
	}
    if(SumDuplicates)
    {
        SparseCommon(data, locsize, total_m, total_n, plus<NT>());
    }
    else
    {
        SparseCommon(data, locsize, total_m, total_n, max<NT>());
    }
}

template <class IT, class NT, class DER>
template <class DELIT>
SpParMat< IT,NT,DER >::SpParMat (const DistEdgeList<DELIT> & DEL, bool removeloops)
{
	commGrid = DEL.commGrid;	
	typedef typename DER::LocalIT LIT;

	int nprocs = commGrid->GetSize();
	int gridrows = commGrid->GetGridRows();
	int gridcols = commGrid->GetGridCols();
	vector< vector<LIT> > data(nprocs);	// enties are pre-converted to local indices before getting pushed into "data"

	LIT m_perproc = DEL.getGlobalV() / gridrows;
	LIT n_perproc = DEL.getGlobalV() / gridcols;

	if(sizeof(LIT) < sizeof(DELIT))
	{
		ostringstream outs;
		outs << "Warning: Using smaller indices for the matrix than DistEdgeList\n";
		outs << "Local matrices are " << m_perproc << "-by-" << n_perproc << endl;
		SpParHelper::Print(outs.str(), commGrid->GetWorld());   // commgrid initialized
	}	
	
    LIT stages = MEM_EFFICIENT_STAGES;		// to lower memory consumption, form sparse matrix in stages
	
	// even if local indices (LIT) are 32-bits, we should work with 64-bits for global info
	int64_t perstage = DEL.nedges / stages;
	LIT totrecv = 0;
	vector<LIT> alledges;
    
	for(LIT s=0; s< stages; ++s)
	{
		int64_t n_befor = s*perstage;
		int64_t n_after= ((s==(stages-1))? DEL.nedges : ((s+1)*perstage));

		// clear the source vertex by setting it to -1
		int realedges = 0;	// these are "local" realedges

		if(DEL.pedges)	
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				int64_t fr = get_v0_from_edge(&(DEL.pedges[i]));
				int64_t to = get_v1_from_edge(&(DEL.pedges[i]));

				if(fr >= 0 && to >= 0)	// otherwise skip
				{
                    IT lrow, lcol;
                    int owner = Owner(DEL.getGlobalV(), DEL.getGlobalV(), fr, to, lrow, lcol);
					data[owner].push_back(lrow);	// row_id
					data[owner].push_back(lcol);	// col_id
					++realedges;
				}
			}
		}
		else
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				if(DEL.edges[2*i+0] >= 0 && DEL.edges[2*i+1] >= 0)	// otherwise skip
				{
                    IT lrow, lcol;
                    int owner = Owner(DEL.getGlobalV(), DEL.getGlobalV(), DEL.edges[2*i+0], DEL.edges[2*i+1], lrow, lcol);
					data[owner].push_back(lrow);
					data[owner].push_back(lcol);
					++realedges;
				}
			}
		}

  		LIT * sendbuf = new LIT[2*realedges];
		int * sendcnt = new int[nprocs];
		int * sdispls = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = data[i].size();

		int * rdispls = new int[nprocs];
		int * recvcnt = new int[nprocs];
		MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT,commGrid->GetWorld()); // share the counts

		sdispls[0] = 0;
		rdispls[0] = 0;
		for(int i=0; i<nprocs-1; ++i)
		{
			sdispls[i+1] = sdispls[i] + sendcnt[i];
			rdispls[i+1] = rdispls[i] + recvcnt[i];
		}
		for(int i=0; i<nprocs; ++i)
			copy(data[i].begin(), data[i].end(), sendbuf+sdispls[i]);
		
		// clear memory
		for(int i=0; i<nprocs; ++i)
			vector<LIT>().swap(data[i]);

		// ABAB: Total number of edges received might not be LIT-addressible
		// However, each edge_id is LIT-addressible
		IT thisrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	// thisrecv = 2*locedges
		LIT * recvbuf = new LIT[thisrecv];
		totrecv += thisrecv;
			
		MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<LIT>(), recvbuf, recvcnt, rdispls, MPIType<LIT>(), commGrid->GetWorld());
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls,sendbuf);
		copy (recvbuf,recvbuf+thisrecv,back_inserter(alledges));	// copy to all edges
		delete [] recvbuf;
	}

	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	LIT locrows, loccols; 
	if(myprocrow != gridrows-1)	locrows = m_perproc;
	else 	locrows = DEL.getGlobalV() - myprocrow * m_perproc;
	if(myproccol != gridcols-1)	loccols = n_perproc;
	else	loccols = DEL.getGlobalV() - myproccol * n_perproc;

  	SpTuples<LIT,NT> A(totrecv/2, locrows, loccols, alledges, removeloops);  	// alledges is empty upon return
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}

template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::RemoveLoops()
{
	MPI_Comm DiagWorld = commGrid->GetDiagWorld();
	IT totrem;
	IT removed = 0;
	if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
	{
		SpTuples<IT,NT> tuples(*spSeq);
		delete spSeq;
		removed  = tuples.RemoveLoops();
		spSeq = new DER(tuples, false);	// Convert to DER
	}
	MPI_Allreduce( &removed, & totrem, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
	return totrem;
}		



template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AddLoops(NT loopval, bool replaceExisting)
{
	MPI_Comm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
	{
    		typedef typename DER::LocalIT LIT;
		SpTuples<LIT,NT> tuples(*spSeq);
		delete spSeq;
		tuples.AddLoops(loopval, replaceExisting);
        	tuples.SortColBased();
		spSeq = new DER(tuples, false);	// Convert to DER
	}
}


// Different values on the diagonal
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AddLoops(FullyDistVec<IT,NT> loopvals, bool replaceExisting)
{
    
    
    if(*loopvals.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::AddLoops() fails!\n", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    if (getncol()!= loopvals.TotalLength())
    {
        SpParHelper::Print("The number of entries in loopvals is not equal to the number of diagonal entries.\n");
        MPI_Abort(MPI_COMM_WORLD,DIMMISMATCH);
    }
    
    // Gather data on the diagonal processor
    IT locsize = loopvals.LocArrSize();
    int rowProcs = commGrid->GetGridCols();
    vector<int> recvcnt(rowProcs, 0);
    vector<int> rdpls(rowProcs, 0);
    MPI_Gather(&locsize, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, commGrid->GetDiagOfProcRow(), commGrid->GetRowWorld());
    partial_sum(recvcnt.data(), recvcnt.data()+rowProcs-1, rdpls.data()+1);

    IT totrecv = rdpls[rowProcs-1] + recvcnt[rowProcs-1];
    vector<NT> rowvals(totrecv);
	MPI_Gatherv(loopvals.arr.data(), locsize, MPIType<NT>(), rowvals.data(), recvcnt.data(), rdpls.data(),
                 MPIType<NT>(), commGrid->GetDiagOfProcRow(), commGrid->GetRowWorld());

   
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
    {
        typedef typename DER::LocalIT LIT;
        SpTuples<LIT,NT> tuples(*spSeq);
        delete spSeq;
        tuples.AddLoops(rowvals, replaceExisting);
        tuples.SortColBased();
        spSeq = new DER(tuples, false);	// Convert to DER
    }
}


//! Pre-allocates buffers for row communication
//! additionally (if GATHERVOPT is defined, incomplete as of March 2016):
//! - Splits the local column indices to sparse & dense pieces to avoid redundant AllGather (sparse pieces get p2p)
template <class IT, class NT, class DER>
template <typename LIT, typename OT>
void SpParMat<IT,NT,DER>::OptimizeForGraph500(OptBuf<LIT,OT> & optbuf)
{
	if(spSeq->getnsplit() > 0)
	{
		SpParHelper::Print("Can not declare preallocated buffers for multithreaded execution\n", commGrid->GetWorld());
		return;
    }

    typedef typename DER::LocalIT LocIT;    // ABAB: should match the type of LIT. Check?
    
    // Set up communication buffers, one for all
	LocIT mA = spSeq->getnrow();
    LocIT nA = spSeq->getncol();
    
	int p_c = commGrid->GetGridCols();
    int p_r = commGrid->GetGridRows();
    
    LocIT rwperproc = mA / p_c; // per processors in row-wise communication
    LocIT cwperproc = nA / p_r; // per processors in column-wise communication
    
#ifdef GATHERVOPT
    LocIT * colinds = seq->GetDCSC()->jc;   // local nonzero column id's
    LocIT locnzc = seq->getnzc();
    LocIT cci = 0;  // index to column id's array (cci: current column index)
    int * gsizes = NULL;
    IT * ents = NULL;
    IT * dpls = NULL;
    vector<LocIT> pack2send;
    
    FullyDistSpVec<IT,IT> dummyRHS ( commGrid, getncol()); // dummy RHS vector to estimate index start position
    IT recveclen;
    
    for(int pid = 1; pid <= p_r; pid++)
    {
        IT diagoffset;
        MPI_Status status;
        IT offset = dummyRHS.RowLenUntil(pid-1);
        int diagneigh = commGrid->GetComplementRank();
        MPI_Sendrecv(&offset, 1, MPIType<IT>(), diagneigh, TRTAGNZ, &diagoffset, 1, MPIType<IT>(), diagneigh, TRTAGNZ, commGrid->GetWorld(), &status);

        LocIT endind = (pid == p_r)? nA : static_cast<LocIT>(pid) * cwperproc;     // the last one might have a larger share (is this fitting to the vector boundaries?)
        while(cci < locnzc && colinds[cci] < endind)
        {
            pack2send.push_back(colinds[cci++]-diagoffset);
        }
        if(pid-1 == myrank) gsizes = new int[p_r];
        MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, pid-1, commGrid->GetColWorld());
        if(pid-1 == myrank)
        {
            IT colcnt = std::accumulate(gsizes, gsizes+p_r, static_cast<IT>(0));
            recvbuf = new IT[colcnt];
            dpls = new IT[p_r]();     // displacements (zero initialized pid)
            std::partial_sum(gsizes, gsizes+p_r-1, dpls+1);
        }
        
        // int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype, void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)
        MPI_Gatherv(SpHelper::p2a(pack2send), mysize, MPIType<LocIT>(), recvbuf, gsizes, dpls, MPIType<LocIT>(), pid-1, commGrid->GetColWorld());
        vector<LocIT>().swap(pack2send);
        
       if(pid-1 == myrank)
       {
           recveclen = dummyRHS.MyLocLength();
           vector< vector<LocIT> > service(recveclen);
           for(int i=0; i< p_r; ++i)
           {
               for(int j=0; j< gsizes[i]; ++j)
               {
                   IT colid2update = recvbuf[dpls[i]+j];
                   if(service[colid2update].size() < GATHERVNEIGHLIMIT)
                   {
                       service.push_back(i);
                   }
                   // else don't increase any further and mark it done after the iterations are complete
               }
           }
       }
    }
#endif

    
	vector<bool> isthere(mA, false); // perhaps the only appropriate use of this crippled data structure
	vector<int> maxlens(p_c,0);	// maximum data size to be sent to any neighbor along the processor row

	for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
	{
		for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
		{
			LocIT rowid = nzit.rowid();
			if(!isthere[rowid])
			{
				LocIT owner = min(nzit.rowid() / rwperproc, (LocIT) p_c-1);
				maxlens[owner]++;
				isthere[rowid] = true;
			}
		}
	}
	SpParHelper::Print("Optimization buffers set\n", commGrid->GetWorld());
	optbuf.Set(maxlens,mA);
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::ActivateThreading(int numsplits)
{
	spSeq->RowSplit(numsplits);
}


/**
 * Parallel routine that returns A*A on the semiring SR
 * Uses only MPI-1 features (relies on simple blocking broadcast)
 **/  
template <class IT, class NT, class DER>
template <typename SR>
void SpParMat<IT,NT,DER>::Square ()
{
	int stages, dummy; 	// last two parameters of productgrid are ignored for synchronous multiplication
	shared_ptr<CommGrid> Grid = ProductGrid(commGrid.get(), commGrid.get(), stages, dummy, dummy);		

	typedef typename DER::LocalIT LIT;
	
	LIT AA_m = spSeq->getnrow();
	LIT AA_n = spSeq->getncol();
	
	DER seqTrn = spSeq->TransposeConst();	// will be automatically discarded after going out of scope		

	MPI_Barrier(commGrid->GetWorld());

	LIT ** NRecvSizes = SpHelper::allocate2D<LIT>(DER::esscount, stages);
	LIT ** TRecvSizes = SpHelper::allocate2D<LIT>(DER::esscount, stages);
	
	SpParHelper::GetSetSizes( *spSeq, NRecvSizes, commGrid->GetRowWorld());
	SpParHelper::GetSetSizes( seqTrn, TRecvSizes, commGrid->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	DER * NRecv; 
	DER * TRecv;
	vector< SpTuples<LIT,NT>  *> tomerge;

	int Nself = commGrid->GetRankInProcRow();
	int Tself = commGrid->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
    {
		vector<LIT> ess;	
		if(i == Nself)  NRecv = spSeq;	// shallow-copy 
		else
		{
			ess.resize(DER::esscount);
			for(int j=0; j< DER::esscount; ++j)
				ess[j] = NRecvSizes[j][i];		// essentials of the ith matrix in this row
			NRecv = new DER();				// first, create the object
		}

		SpParHelper::BCastMatrix(Grid->GetRowWorld(), *NRecv, ess, i);	// then, broadcast its elements	
		ess.clear();	
		
		if(i == Tself)  TRecv = &seqTrn;	// shallow-copy
		else
		{
			ess.resize(DER::esscount);		
			for(int j=0; j< DER::esscount; ++j)
				ess[j] = TRecvSizes[j][i];
			TRecv = new DER();
		}
		SpParHelper::BCastMatrix(Grid->GetColWorld(), *TRecv, ess, i);	

		SpTuples<LIT,NT> * AA_cont = MultiplyReturnTuples<SR, NT>(*NRecv, *TRecv, false, true);
		if(!AA_cont->isZero()) 
			tomerge.push_back(AA_cont);

		if(i != Nself)	delete NRecv;
		if(i != Tself)  delete TRecv;
	}

	SpHelper::deallocate2D(NRecvSizes, DER::esscount);
	SpHelper::deallocate2D(TRecvSizes, DER::esscount);
	
	delete spSeq;		
	spSeq = new DER(MergeAll<SR>(tomerge, AA_m, AA_n), false);	// First get the result in SpTuples, then convert to UDER
	for(unsigned int i=0; i<tomerge.size(); ++i)
		delete tomerge[i];
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Transpose()
{
	if(commGrid->myproccol == commGrid->myprocrow)	// Diagonal
	{
		spSeq->Transpose();			
	}
	else
	{
		typedef typename DER::LocalIT LIT;
		SpTuples<LIT,NT> Atuples(*spSeq);
		LIT locnnz = Atuples.getnnz();
		LIT * rows = new LIT[locnnz];
		LIT * cols = new LIT[locnnz];
		NT * vals = new NT[locnnz];
		for(LIT i=0; i < locnnz; ++i)
		{
			rows[i] = Atuples.colindex(i);	// swap (i,j) here
			cols[i] = Atuples.rowindex(i);
			vals[i] = Atuples.numvalue(i);
		}
		LIT locm = getlocalcols();
		LIT locn = getlocalrows();
		delete spSeq;

		LIT remotem, remoten, remotennz;
		swap(locm,locn);
		int diagneigh = commGrid->GetComplementRank();

		MPI_Status status;
		MPI_Sendrecv(&locnnz, 1, MPIType<LIT>(), diagneigh, TRTAGNZ, &remotennz, 1, MPIType<LIT>(), diagneigh, TRTAGNZ, commGrid->GetWorld(), &status);
		MPI_Sendrecv(&locn, 1, MPIType<LIT>(), diagneigh, TRTAGM, &remotem, 1, MPIType<LIT>(), diagneigh, TRTAGM, commGrid->GetWorld(), &status);
		MPI_Sendrecv(&locm, 1, MPIType<LIT>(), diagneigh, TRTAGN, &remoten, 1, MPIType<LIT>(), diagneigh, TRTAGN, commGrid->GetWorld(), &status);

		LIT * rowsrecv = new LIT[remotennz];
		MPI_Sendrecv(rows, locnnz, MPIType<LIT>(), diagneigh, TRTAGROWS, rowsrecv, remotennz, MPIType<LIT>(), diagneigh, TRTAGROWS, commGrid->GetWorld(), &status);
		delete [] rows;

		LIT * colsrecv = new LIT[remotennz];
		MPI_Sendrecv(cols, locnnz, MPIType<LIT>(), diagneigh, TRTAGCOLS, colsrecv, remotennz, MPIType<LIT>(), diagneigh, TRTAGCOLS, commGrid->GetWorld(), &status);
		delete [] cols;

		NT * valsrecv = new NT[remotennz];
		MPI_Sendrecv(vals, locnnz, MPIType<NT>(), diagneigh, TRTAGVALS, valsrecv, remotennz, MPIType<NT>(), diagneigh, TRTAGVALS, commGrid->GetWorld(), &status);
		delete [] vals;

		tuple<LIT,LIT,NT> * arrtuples = new tuple<LIT,LIT,NT>[remotennz];
		for(LIT i=0; i< remotennz; ++i)
		{
			arrtuples[i] = make_tuple(rowsrecv[i], colsrecv[i], valsrecv[i]);
		}	
		DeleteAll(rowsrecv, colsrecv, valsrecv);
		ColLexiCompare<LIT,NT> collexicogcmp;
		sort(arrtuples , arrtuples+remotennz, collexicogcmp );	// sort w.r.t columns here

		spSeq = new DER();
		spSeq->Create( remotennz, remotem, remoten, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create
	}	
}		


template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat< IT,NT,DER >::SaveGathered(string filename, HANDLER handler, bool transpose) const
{
	int proccols = commGrid->GetGridCols();
	int procrows = commGrid->GetGridRows();
	IT totalm = getnrow();
	IT totaln = getncol();
	IT totnnz = getnnz();
	int flinelen = 0;
	ofstream out;
	if(commGrid->GetRank() == 0)
	{
		std::string s;
		std::stringstream strm;
		strm << "%%MatrixMarket matrix coordinate real general" << endl;
		strm << totalm << " " << totaln << " " << totnnz << endl;
		s = strm.str();
		out.open(filename.c_str(),ios_base::trunc);
		flinelen = s.length();
		out.write(s.c_str(), flinelen);
		out.close();
	}
	int colrank = commGrid->GetRankInProcCol(); 
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	locnrows[colrank] = (IT) getlocalrows();
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	IT roffset = accumulate(locnrows, locnrows+colrank, 0);
	delete [] locnrows;	

	MPI_Datatype datatype;
	MPI_Type_contiguous(sizeof(pair<IT,NT>), MPI_CHAR, &datatype);
	MPI_Type_commit(&datatype);

	for(int i = 0; i < procrows; i++)	// for all processor row (in order)
	{
		if(commGrid->GetRankInProcCol() == i)	// only the ith processor row
		{ 
			IT localrows = spSeq->getnrow();    // same along the processor row
			vector< vector< pair<IT,NT> > > csr(localrows);
			if(commGrid->GetRankInProcRow() == 0)	// get the head of processor row 
			{
				IT localcols = spSeq->getncol();    // might be different on the last processor on this processor row
				MPI_Bcast(&localcols, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
					{
						csr[nzit.rowid()].push_back( make_pair(colit.colid(), nzit.value()) );
					}
				}
			}
			else	// get the rest of the processors
			{
				IT n_perproc;
				MPI_Bcast(&n_perproc, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
				IT noffset = commGrid->GetRankInProcRow() * n_perproc; 
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
					{
						csr[nzit.rowid()].push_back( make_pair(colit.colid() + noffset, nzit.value()) );
					}
				}
			}
			pair<IT,NT> * ents = NULL;
			int * gsizes = NULL, * dpls = NULL;
			if(commGrid->GetRankInProcRow() == 0)	// only the head of processor row 
			{
				out.open(filename.c_str(),std::ios_base::app);
				gsizes = new int[proccols];
				dpls = new int[proccols]();	// displacements (zero initialized pid) 
			}
			for(int j = 0; j < localrows; ++j)	
			{
				IT rowcnt = 0;
				sort(csr[j].begin(), csr[j].end());
				int mysize = csr[j].size();
				MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, 0, commGrid->GetRowWorld());
				if(commGrid->GetRankInProcRow() == 0)	
				{
					rowcnt = std::accumulate(gsizes, gsizes+proccols, static_cast<IT>(0));
					std::partial_sum(gsizes, gsizes+proccols-1, dpls+1);
					ents = new pair<IT,NT>[rowcnt];	// nonzero entries in the j'th local row
				}

				// int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype, 
				// 		    void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)	
				MPI_Gatherv(SpHelper::p2a(csr[j]), mysize, datatype, ents, gsizes, dpls, datatype, 0, commGrid->GetRowWorld());
				if(commGrid->GetRankInProcRow() == 0)	
				{
					for(int k=0; k< rowcnt; ++k)
					{
						//out << j + roffset + 1 << "\t" << ents[k].first + 1 <<"\t" << ents[k].second << endl;
						if (!transpose)
							// regular
							out << j + roffset + 1 << "\t" << ents[k].first + 1 << "\t";
						else
							// transpose row/column
							out << ents[k].first + 1 << "\t" << j + roffset + 1 << "\t";
						handler.save(out, ents[k].second, j + roffset, ents[k].first);
						out << endl;
					}
					delete [] ents;
				}
			}
			if(commGrid->GetRankInProcRow() == 0)
			{
				DeleteAll(gsizes, dpls);
				out.close();
			}
		} // end_if the ith processor row 
		MPI_Barrier(commGrid->GetWorld());		// signal the end of ith processor row iteration (so that all processors block)
	}
}



//! Handles all sorts of orderings as long as there are no duplicates
//! Requires proper matrix market banner at the moment
//! Might replace ReadDistribute in the long term
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
void SpParMat< IT,NT,DER >::ParallelReadMM (const string & filename, bool onebased, _BinaryOperation BinOp)
{
    int32_t type = -1;
    int32_t symmetric = 0;
    int64_t nrows, ncols, nonzeros;
    int64_t linesread = 0;
    
    FILE *f;
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    if(myrank == 0)
    {
        MM_typecode matcode;
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            printf("COMBBLAS: Matrix-market file %s can not be found\n", filename.c_str());
            MPI_Abort(MPI_COMM_WORLD, NOFILE);
        }
        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }
        linesread++;
        
        if (mm_is_complex(matcode))
        {
            printf("Sorry, this application does not support complext types");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        }
        else if(mm_is_real(matcode))
        {
            cout << "Matrix is Float" << endl;
            type = 0;
        }
        else if(mm_is_integer(matcode))
        {
            cout << "Matrix is Integer" << endl;
            type = 1;
        }
        else if(mm_is_pattern(matcode))
        {
            cout << "Matrix is Boolean" << endl;
            type = 2;
        }
        if(mm_is_symmetric(matcode))
        {
            cout << "Matrix is symmetric" << endl;
            symmetric = 1;
        }
        int ret_code;
        if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nonzeros, &linesread)) !=0)  // ABAB: mm_read_mtx_crd_size made 64-bit friendly
            exit(1);
    
        cout << "Total number of nonzeros expected across all processors is " << nonzeros << endl;

    }
    MPI_Bcast(&type, 1, MPI_INT, 0, commGrid->commWorld);
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, commGrid->commWorld);
    MPI_Bcast(&nrows, 1, MPIType<int64_t>(), 0, commGrid->commWorld);
    MPI_Bcast(&ncols, 1, MPIType<int64_t>(), 0, commGrid->commWorld);
    MPI_Bcast(&nonzeros, 1, MPIType<int64_t>(), 0, commGrid->commWorld);

    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }
    int64_t file_size = st.st_size;
    MPI_Offset fpos, end_fpos;
    if(commGrid->GetRank() == 0)    // the offset needs to be for this rank
    {
        cout << "File is " << file_size << " bytes" << endl;
        fpos = ftell(f);
        fclose(f);
    }
    else
    {
        fpos = myrank * file_size / nprocs;

    }
    if(myrank != (nprocs-1)) end_fpos = (myrank + 1) * file_size / nprocs;
    else end_fpos = file_size;

    MPI_File mpi_fh;
    MPI_File_open (commGrid->commWorld, const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);

	 
    typedef typename DER::LocalIT LIT;
    vector<LIT> rows;
    vector<LIT> cols;
    vector<NT> vals;

    vector<string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
    SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    MPI_Barrier(commGrid->commWorld);

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
        SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, commGrid->commWorld);
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        cout << "Reading finished. Total number of entries read across all processors is " << allentriesread << endl;
#endif

    vector< vector < tuple<LIT,LIT,NT> > > data(nprocs);
    
    LIT locsize = rows.size();   // remember: locsize != entriesread (unless the matrix is unsymmetric)
    for(LIT i=0; i<locsize; ++i)
    {
        LIT lrow, lcol;
        int owner = Owner(nrows, ncols, rows[i], cols[i], lrow, lcol);
        data[owner].push_back(make_tuple(lrow,lcol,vals[i]));
    }
    vector<LIT>().swap(rows);
    vector<LIT>().swap(cols);
    vector<NT>().swap(vals);	

#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        cout << "Packing to recepients finished, about to send..." << endl;
#endif
    
    if(spSeq)   delete spSeq;
    SparseCommon(data, locsize, nrows, ncols, BinOp);
}


//! Handles all sorts of orderings as long as there are no duplicates
//! May perform better when the data is already reverse column-sorted (i.e. in decreasing order)
//! if nonum is true, then numerics are not supplied and they are assumed to be all 1's
template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat< IT,NT,DER >::ReadDistribute (const string & filename, int master, bool nonum, HANDLER handler, bool transpose, bool pario)
{
#ifdef TAU_PROFILE
   	TAU_PROFILE_TIMER(rdtimer, "ReadDistribute", "void SpParMat::ReadDistribute (const string & , int, bool, HANDLER, bool)", TAU_DEFAULT);
   	TAU_PROFILE_START(rdtimer);
#endif

	ifstream infile;
	FILE * binfile = NULL;	// points to "past header" if the file is binary
	int seeklength = 0;
	HeaderInfo hfile;
	if(commGrid->GetRank() == master)	// 1 processor
	{
		hfile = ParseHeader(filename, binfile, seeklength);
	}
	MPI_Bcast(&seeklength, 1, MPI_INT, master, commGrid->commWorld);

	IT total_m, total_n, total_nnz;
	IT m_perproc = 0, n_perproc = 0;

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (2 * sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (2 * sizeof(IT) + sizeof(NT)));
	if(pario)
	{
		// since all colneighs will be reading the data at the same time
		// chances are they might all read the data that should go to one
		// in that case buffperrowneigh > colneighs * buffpercolneigh 
		// in order not to overflow
		buffpercolneigh /= colneighs; 
		if(seeklength == 0)
			SpParHelper::Print("COMBBLAS: Parallel I/O requested but binary header is corrupted\n", commGrid->GetWorld());
	}

	// make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);
	if(std::max(buffpercolneigh * colneighs, buffperrowneigh * rowneighs) > numeric_limits<int>::max())
	{  
		SpParHelper::Print("COMBBLAS: MPI doesn't support sending int64_t send/recv counts or displacements\n", commGrid->GetWorld());
	}
 
	int * cdispls = new int[colneighs];
	for (IT i=0; i<colneighs; ++i)  cdispls[i] = i*buffpercolneigh;
	int * rdispls = new int[rowneighs];
	for (IT i=0; i<rowneighs; ++i)  rdispls[i] = i*buffperrowneigh;		

	int *ccurptrs = NULL, *rcurptrs = NULL;	
	int recvcount = 0;
	IT * rows = NULL; 
	IT * cols = NULL;
	NT * vals = NULL;

	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);	
	vector< tuple<IT, IT, NT> > localtuples;

	if(commGrid->GetRank() == master)	// 1 processor
	{		
		if( !hfile.fileexists )
		{
			SpParHelper::Print( "COMBBLAS: Input file doesn't exist\n", commGrid->GetWorld());
			total_n = 0; total_m = 0;	
			BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
			return;
		}
		if (hfile.headerexists && hfile.format == 1) 
		{
			SpParHelper::Print("COMBBLAS: Ascii input with binary headers is not supported\n", commGrid->GetWorld());
			total_n = 0; total_m = 0;	
			BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
			return;
		}
		if ( !hfile.headerexists )	// no header - ascii file (at this point, file exists)
		{
			infile.open(filename.c_str());
			char comment[256];
			infile.getline(comment,256);
			while(comment[0] == '%')
			{
				infile.getline(comment,256);
			}
			stringstream ss;
			ss << string(comment);
			ss >> total_m >> total_n >> total_nnz;
			if(pario)
			{
				SpParHelper::Print("COMBBLAS: Trying to read binary headerless file in parallel, aborting\n", commGrid->GetWorld());
				total_n = 0; total_m = 0;	
				BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
				return;				
			}
		}
		else // hfile.headerexists && hfile.format == 0
		{
			total_m = hfile.m;
			total_n = hfile.n;
			total_nnz = hfile.nnz;
		}
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
		AllocateSetBuffers(rows, cols, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);

		if(seeklength > 0 && pario)   // sqrt(p) processors also do parallel binary i/o
		{
			IT entriestoread =  total_nnz / colneighs;
			#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " entries to read: " << entriestoread << endl;
			oput.close();
			#endif
			ReadAllMine(binfile, rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
				rowneighs, colneighs, buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow, transpose);
		}
		else	// only this (master) is doing I/O (text or binary)
		{
			IT temprow, tempcol;
			NT tempval;	
			IT ntrow = 0, ntcol = 0; // not transposed row and column index
			char line[1024];
			bool nonumline = nonum;
			IT cnz = 0;
			for(; cnz < total_nnz; ++cnz)
			{	
				int colrec;
				size_t commonindex;
				stringstream linestream;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					// read one line at a time so that missing numerical values can be detected
					infile.getline(line, 1024);
					linestream << line;
					linestream >> temprow >> tempcol;
					if (!nonum)
					{
						// see if this line has a value
						linestream >> skipws;
						nonumline = linestream.eof();
					}
					--temprow;	// file is 1-based where C-arrays are 0-based
					--tempcol;
					ntrow = temprow;
					ntcol = tempcol;
				}
				else if(hfile.headerexists && (!feof(binfile)) ) 
				{
					handler.binaryfill(binfile, temprow , tempcol, tempval);
				}
				if (transpose)
				{
					IT swap = temprow;
					temprow = tempcol;
					tempcol = swap;
				}
				colrec = std::min(static_cast<int>(temprow / m_perproc), colneighs-1);	// precipient processor along the column
				commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
					
				rows[ commonindex ] = temprow;
				cols[ commonindex ] = tempcol;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					vals[ commonindex ] = nonumline ? handler.getNoNum(ntrow, ntcol) : handler.read(linestream, ntrow, ntcol); //tempval;
				}
				else if(hfile.headerexists && (!feof(binfile)) ) 
				{
					vals[ commonindex ] = tempval;
				}
				++ (ccurptrs[colrec]);				
				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld); // first, send the receive counts

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * temprows = new IT[recvcount];
					IT * tempcols = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					MPI_Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
					MPI_Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
					MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->colWorld);

					fill_n(ccurptrs, colneighs, 0);  				// finally, reset current pointers !
					DeleteAll(rows, cols, vals);
					
					HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, 
							buffperrowneigh, rowneighs, recvcount, m_perproc, n_perproc, rankinrow);
					
					if( cnz != (total_nnz-1) )	// otherwise the loop will exit with noone to claim memory back
					{
						// reuse these buffers for the next vertical communication								
						rows = new IT [ buffpercolneigh * colneighs ];
						cols = new IT [ buffpercolneigh * colneighs ];
						vals = new NT [ buffpercolneigh * colneighs ];
					}
				} // end_if for "send buffer is full" case 
			} // end_for for "cnz < entriestoread" case
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			fill_n(ccurptrs, colneighs, numeric_limits<int>::max());	
			MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld);

			// And along the row ...
			fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());				
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
		}	// end of "else" (only one processor reads) block
	}	// end_if for "master processor" case
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;

		if(seeklength > 0 && pario)   // these processors also do parallel binary i/o
		{
			binfile = fopen(filename.c_str(), "rb");
			IT entrysize = handler.entrylength();
			int myrankincol = commGrid->GetRankInProcCol();
			IT perreader = total_nnz / colneighs;
			IT read_offset = entrysize * static_cast<IT>(myrankincol) * perreader + seeklength;
			IT entriestoread = perreader;
			if (myrankincol == colneighs-1) 
				entriestoread = total_nnz - static_cast<IT>(myrankincol) * perreader;
			fseek(binfile, read_offset, SEEK_SET);

			#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " OFFSET : " << read_offset << " entries to read: " << entriestoread << endl;
			oput.close();
			#endif
			
			AllocateSetBuffers(rows, cols, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);
			ReadAllMine(binfile, rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
				rowneighs, colneighs, buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow, transpose);
		}
		else // only master does the I/O
		{
			while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
			{
				// void MPI::Comm::Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], const MPI::Datatype& sendtype,
				//				void* recvbuf, int recvcount, const MPI::Datatype & recvtype, int root) const
				// The outcome is as if the root executed n send operations,
				//	MPI_Send(sendbuf + displs[i] * extent(sendtype), sendcounts[i], sendtype, i, ...)
				// and each process executed a receive,
				// 	MPI_Recv(recvbuf, recvcount, recvtype, root, ...)
				// The send buffer is ignored for all nonroot processes.
				
				MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld);                       // first receive the receive counts ...
				if( recvcount == numeric_limits<int>::max()) break;
				
				// create space for incoming data ... 
				IT * temprows = new IT[recvcount];
				IT * tempcols = new IT[recvcount];
				NT * tempvals = new NT[recvcount];
				
				// receive actual data ... (first 4 arguments are ignored in the receiver side)
				MPI_Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
				MPI_Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
				MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->colWorld);

				// now, send the data along the horizontal
				rcurptrs = new int[rowneighs];
				fill_n(rcurptrs, rowneighs, 0);	
				
				// HorizontalSend frees the memory of temp_xxx arrays and then creates and frees memory of all the six arrays itself
				HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, 
					buffperrowneigh, rowneighs, recvcount, m_perproc, n_perproc, rankinrow);
			}
		}
		
		// Signal the end of file to other processors along the row
		fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());				
		MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
		delete [] rcurptrs;	
	}
	else		// r * (s-1) processors that only participate in the horizontal communication step
	{
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
		{
			// receive the receive count
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
			if( recvcount == numeric_limits<int>::max())
				break;

			// create space for incoming data ... 
			IT * temprows = new IT[recvcount];
			IT * tempcols = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			MPI_Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
			MPI_Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
			MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->rowWorld);

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			
			for(IT i=0; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
			DeleteAll(temprows,tempcols,tempvals);
		}
	}
	DeleteAll(cdispls, rdispls);
	tuple<IT,IT,NT> * arrtuples = new tuple<IT,IT,NT>[localtuples.size()];  // the vector will go out of scope, make it stick !
	copy(localtuples.begin(), localtuples.end(), arrtuples);

 	IT localm = (commGrid->myprocrow != (commGrid->grrows-1))? m_perproc: (total_m - (m_perproc * (commGrid->grrows-1)));
 	IT localn = (commGrid->myproccol != (commGrid->grcols-1))? n_perproc: (total_n - (n_perproc * (commGrid->grcols-1)));
	spSeq->Create( localtuples.size(), localm, localn, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create

#ifdef TAU_PROFILE
   	TAU_PROFILE_STOP(rdtimer);
#endif
	return;
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AllocateSetBuffers(IT * & rows, IT * & cols, NT * & vals,  int * & rcurptrs, int * & ccurptrs, int rowneighs, int colneighs, IT buffpercolneigh)
{
	// allocate buffers on the heap as stack space is usually limited
	rows = new IT [ buffpercolneigh * colneighs ];
	cols = new IT [ buffpercolneigh * colneighs ];
	vals = new NT [ buffpercolneigh * colneighs ];
	
	ccurptrs = new int[colneighs];
	rcurptrs = new int[rowneighs];
	fill_n(ccurptrs, colneighs, 0);	// fill with zero
	fill_n(rcurptrs, rowneighs, 0);	
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::BcastEssentials(MPI_Comm & world, IT & total_m, IT & total_n, IT & total_nnz, int master)
{
	MPI_Bcast(&total_m, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_n, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_nnz, 1, MPIType<IT>(), master, world);
}
	
/*
 * @post {rows, cols, vals are pre-allocated on the heap after this call} 
 * @post {ccurptrs are set to zero; so that if another call is made to this function without modifying ccurptrs, no data will be send from this procesor}
 */
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::VerticalSend(IT * & rows, IT * & cols, NT * & vals, vector< tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
				  IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, int rankinrow)
{
	// first, send/recv the counts ...
	int * colrecvdispls = new int[colneighs];
	int * colrecvcounts = new int[colneighs];
	MPI_Alltoall(ccurptrs, 1, MPI_INT, colrecvcounts, 1, MPI_INT, commGrid->colWorld);      // share the request counts
	int totrecv = accumulate(colrecvcounts,colrecvcounts+colneighs,0);	
	colrecvdispls[0] = 0; 		// receive displacements are exact whereas send displacements have slack
	for(int i=0; i<colneighs-1; ++i)
		colrecvdispls[i+1] = colrecvdispls[i] + colrecvcounts[i];
	
	// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
	IT * temprows = new IT[totrecv];
	IT * tempcols = new IT[totrecv];
	NT * tempvals = new NT[totrecv];
	
	// then, exchange all buffers that to their recipients ...
	MPI_Alltoallv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, colrecvcounts, colrecvdispls, MPIType<IT>(), commGrid->colWorld);
	MPI_Alltoallv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, colrecvcounts, colrecvdispls, MPIType<IT>(), commGrid->colWorld);
	MPI_Alltoallv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, colrecvcounts, colrecvdispls, MPIType<NT>(), commGrid->colWorld);

	// finally, reset current pointers !
	fill_n(ccurptrs, colneighs, 0);
	DeleteAll(colrecvdispls, colrecvcounts);
	DeleteAll(rows, cols, vals);
	
	// rcurptrs/rdispls are zero initialized scratch space
	HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, buffperrowneigh, rowneighs, totrecv, m_perproc, n_perproc, rankinrow);
	
	// reuse these buffers for the next vertical communication								
	rows = new IT [ buffpercolneigh * colneighs ];
	cols = new IT [ buffpercolneigh * colneighs ];
	vals = new NT [ buffpercolneigh * colneighs ];
}


/**
 * Private subroutine of ReadDistribute. 
 * Executed by p_r processors on the first processor column. 
 * @pre {rows, cols, vals are pre-allocated on the heap before this call} 
 * @param[in] rankinrow {row head's rank in its processor row - determines the scatter person} 
 */
template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat<IT,NT,DER>::ReadAllMine(FILE * binfile, IT * & rows, IT * & cols, NT * & vals, vector< tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
		IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, IT entriestoread, HANDLER handler, int rankinrow, bool transpose)
{
	assert(entriestoread != 0);
	IT cnz = 0;
	IT temprow, tempcol;
	NT tempval;
	int finishedglobal = 1;
	while(cnz < entriestoread && !feof(binfile))	// this loop will execute at least once
	{
		handler.binaryfill(binfile, temprow , tempcol, tempval);
		if (transpose)
		{
			IT swap = temprow;
			temprow = tempcol;
			tempcol = swap;
		}
		int colrec = std::min(static_cast<int>(temprow / m_perproc), colneighs-1);	// precipient processor along the column
		size_t commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
		rows[ commonindex ] = temprow;
		cols[ commonindex ] = tempcol;
		vals[ commonindex ] = tempval;
		++ (ccurptrs[colrec]);	
		if(ccurptrs[colrec] == buffpercolneigh || (cnz == (entriestoread-1)) )		// one buffer is full, or this processor's share is done !
		{			
			#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "To column neighbors: ";
			copy(ccurptrs, ccurptrs+colneighs, ostream_iterator<int>(oput, " ")); oput << endl;
			oput.close();
			#endif

			VerticalSend(rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
					rowneighs, colneighs, buffperrowneigh, buffpercolneigh, rankinrow);

			if(cnz == (entriestoread-1))	// last execution of the outer loop
			{
				int finishedlocal = 1;	// I am done, but let me check others 
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
				while(!finishedglobal)
				{
					#ifdef DEBUG
					ofstream oput;
					commGrid->OpenDebugFile("Read", oput);
					oput << "To column neighbors: ";
					copy(ccurptrs, ccurptrs+colneighs, ostream_iterator<int>(oput, " ")); oput << endl;
					oput.close();
					#endif

					// postcondition of VerticalSend: ccurptrs are set to zero
					// if another call is made to this function without modifying ccurptrs, no data will be send from this procesor
					VerticalSend(rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
						rowneighs, colneighs, buffperrowneigh, buffpercolneigh, rankinrow);

					MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
				}
			}
			else // the other loop will continue executing
			{
				int finishedlocal = 0;
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
			}
		} // end_if for "send buffer is full" case 
		++cnz;
	}

	// signal the end to row neighbors
	fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());				
	int recvcount;
	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
}


/**
 * Private subroutine of ReadDistribute
 * @param[in] rankinrow {Row head's rank in its processor row}
 * Initially temp_xxx arrays carry data received along the proc. column AND needs to be sent along the proc. row
 * After usage, function frees the memory of temp_xxx arrays and then creates and frees memory of all the six arrays itself
 */
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::HorizontalSend(IT * & rows, IT * & cols, NT * & vals, IT * & temprows, IT * & tempcols, NT * & tempvals, vector < tuple <IT,IT,NT> > & localtuples, 
					 int * rcurptrs, int * rdispls, IT buffperrowneigh, int rowneighs, int recvcount, IT m_perproc, IT n_perproc, int rankinrow)
{	
	rows = new IT [ buffperrowneigh * rowneighs ];
	cols = new IT [ buffperrowneigh * rowneighs ];
	vals = new NT [ buffperrowneigh * rowneighs ];
	
	// prepare to send the data along the horizontal
	for(int i=0; i< recvcount; ++i)
	{
		int rowrec = std::min(static_cast<int>(tempcols[i] / n_perproc), rowneighs-1);
		rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
		cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
		vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
		++ (rcurptrs[rowrec]);	
	}

	#ifdef IODEBUG
	ofstream oput;
	commGrid->OpenDebugFile("Read", oput);
	oput << "To row neighbors: ";
	copy(rcurptrs, rcurptrs+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
	oput << "Row displacements were: ";
	copy(rdispls, rdispls+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
	oput.close();
	#endif

	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld); // Send the receive counts for horizontal communication	

	// the data is now stored in rows/cols/vals, can reset temporaries
	// sets size and capacity to new recvcount
	DeleteAll(temprows, tempcols, tempvals);
	temprows = new IT[recvcount];
	tempcols = new IT[recvcount];
	tempvals = new NT[recvcount];
	
	// then, send all buffers that to their recipients ...
	MPI_Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
	MPI_Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
	MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->rowWorld);

	// now push what is ours to tuples
	IT moffset = commGrid->myprocrow * m_perproc; 
	IT noffset = commGrid->myproccol * n_perproc; 
	
	for(int i=0; i< recvcount; ++i)
	{					
		localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
	}
	
	fill_n(rcurptrs, rowneighs, 0);
	DeleteAll(rows, cols, vals, temprows, tempcols, tempvals);		
}


//! The input parameters' identity (zero) elements as well as 
//! their communication grid is preserved while outputting
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Find (FullyDistVec<IT,IT> & distrows, FullyDistVec<IT,IT> & distcols, FullyDistVec<IT,NT> & distvals) const
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Find() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0); 
	FullyDistVec<IT,NT> nvals ( distvals.commGrid, globallen, NT()); 
	
	IT prelen = Atuples.getnnz();
	//IT postlen = nrows.MyLocLength();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT prelenuntil = accumulate(prelens, prelens+rank, static_cast<IT>(0));

	int * sendcnt = new int[nprocs]();	// zero initialize
	IT * rows = new IT[prelen];
	IT * cols = new IT[prelen];
	NT * vals = new NT[prelen];

	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol(); 
	int rowneighs = commGrid->GetGridCols();
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	IT * locncols = new IT[rowneighs];
	locnrows[colrank] = getlocalrows();
	locncols[rowrank] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetRowWorld());

	IT roffset = accumulate(locnrows, locnrows+colrank, static_cast<IT>(0));
	IT coffset = accumulate(locncols, locncols+rowrank, static_cast<IT>(0));
	
	DeleteAll(locnrows, locncols);
	for(int i=0; i< prelen; ++i)
	{
		IT locid;	// ignore local id, data will come in order
		int owner = nrows.Owner(prelenuntil+i, locid);
		sendcnt[owner]++;

		rows[i] = Atuples.rowindex(i) + roffset;	// need the global row index
		cols[i] = Atuples.colindex(i) + coffset;	// need the global col index
		vals[i] = Atuples.numvalue(i);
	}

	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());   // get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	MPI_Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(nrows.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(ncols.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(vals, sendcnt, sdpls, MPIType<NT>(), SpHelper::p2a(nvals.arr), recvcnt, rdpls, MPIType<NT>(), commGrid->GetWorld());

	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	DeleteAll(prelens, rows, cols, vals);
	distrows = nrows;
	distcols = ncols;
	distvals = nvals;
}

//! The input parameters' identity (zero) elements as well as 
//! their communication grid is preserved while outputting
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Find (FullyDistVec<IT,IT> & distrows, FullyDistVec<IT,IT> & distcols) const
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) )
	{
		SpParHelper::Print("Grids are not comparable, Find() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0); 
	
	IT prelen = Atuples.getnnz();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT prelenuntil = accumulate(prelens, prelens+rank, static_cast<IT>(0));

	int * sendcnt = new int[nprocs]();	// zero initialize
	IT * rows = new IT[prelen];
	IT * cols = new IT[prelen];
	NT * vals = new NT[prelen];

	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol(); 
	int rowneighs = commGrid->GetGridCols();
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	IT * locncols = new IT[rowneighs];
	locnrows[colrank] = getlocalrows();
	locncols[rowrank] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetColWorld());
	IT roffset = accumulate(locnrows, locnrows+colrank, static_cast<IT>(0));
	IT coffset = accumulate(locncols, locncols+rowrank, static_cast<IT>(0));
	
	DeleteAll(locnrows, locncols);
	for(int i=0; i< prelen; ++i)
	{
		IT locid;	// ignore local id, data will come in order
		int owner = nrows.Owner(prelenuntil+i, locid);
		sendcnt[owner]++;

		rows[i] = Atuples.rowindex(i) + roffset;	// need the global row index
		cols[i] = Atuples.colindex(i) + coffset;	// need the global col index
	}

	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());   // get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	MPI_Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(nrows.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(ncols.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());

	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	DeleteAll(prelens, rows, cols, vals);
	distrows = nrows;
	distcols = ncols;
}

template <class IT, class NT, class DER>
ofstream& SpParMat<IT,NT,DER>::put(ofstream& outfile) const
{
	outfile << (*spSeq) << endl;
	return outfile;
}

template <class IU, class NU, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMat<IU, NU, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}

/**
  * @param[in] grow {global row index}
  * @param[in] gcol {global column index}
  * @param[out] lrow {row index local to the owner}
  * @param[out] lcol {col index local to the owner}
  * @returns {owner processor id}
 **/
template <class IT, class NT,class DER>
template <typename LIT>
int SpParMat<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const
{
	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	IT m_perproc = total_m / procrows;
	IT n_perproc = total_n / proccols;

	int own_procrow;	// owner's processor row
	if(m_perproc != 0)
	{
		own_procrow = std::min(static_cast<int>(grow / m_perproc), procrows-1);	// owner's processor row
	}
	else	// all owned by the last processor row
	{
		own_procrow = procrows -1;
	}
	int own_proccol;
	if(n_perproc != 0)
	{
		own_proccol = std::min(static_cast<int>(gcol / n_perproc), proccols-1);
	}
	else
	{
		own_proccol = proccols-1;
	}
	lrow = grow - (own_procrow * m_perproc);
	lcol = gcol - (own_proccol * n_perproc);
	return commGrid->GetRank(own_procrow, own_proccol);
}

/**
  * @param[out] rowOffset {Row offset imposed by process grid. Global row index = rowOffset + local row index.}
  * @param[out] colOffset {Column offset imposed by process grid. Global column index = colOffset + local column index.}
 **/
template <class IT, class NT,class DER>
void SpParMat<IT,NT,DER>::GetPlaceInGlobalGrid(IT& rowOffset, IT& colOffset) const
{
	IT total_rows = getnrow();
	IT total_cols = getncol();

	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	IT rows_perproc = total_rows / procrows;
	IT cols_perproc = total_cols / proccols;
	
	rowOffset = commGrid->GetRankInProcCol()*rows_perproc;
	colOffset = commGrid->GetRankInProcRow()*cols_perproc;
}
	