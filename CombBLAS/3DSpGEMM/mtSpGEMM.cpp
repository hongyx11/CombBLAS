#include <cstdlib>
#include <parallel/algorithm>
#include "../CombBLAS.h"


/*
template<class IT, class NT>
void FillColInds4(const IT * colnums, IT nind, vector< pair<IT,IT> > & colinds, IT * aux, IT csize)
{
    bool found;
    for(IT j =0; j< nind; ++j)
    {
        IT pos = AuxIndex(colnums[j], found, aux, csize);
        if(found)
        {
            colinds[j].first = cp[pos];
            colinds[j].second = cp[pos+1];
        }
        else 	// not found, signal by setting first = second
        {
            colinds[j].first = 0;
            colinds[j].second = 0;
        }
    }
}
*/

template <typename T>
T* prefixsum(T* in, int size, int nthreads)
{
    vector<T> tsum(nthreads+1);
    tsum[0] = 0;
    T* out = new T[size+1];
    out[0] = 0;
    T* psum = &out[1];
    
#pragma omp parallel
    {
        int ithread = omp_get_thread_num();
        T sum = 0;
#pragma omp for schedule(static)
        for (int i=0; i<size; i++)
        {
            sum += in[i];
            psum[i] = sum;
        }
        
        tsum[ithread+1] = sum;
#pragma omp barrier
        T offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += tsum[i];
        }
#pragma omp for schedule(static)
        for (int i=0; i<size; i++)
        {
            psum[i] += offset;
        }
    
    }
    return out;
}




// multithreaded
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    //double t01 = MPI_Wtime();
  
    
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IT, NTO>(0, mdim, ndim);
    }
    
    Dcsc<IT,NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT,NT2>* Bdcsc = B.GetDCSC();
    
    IT nA = A.getncol();
    IT cnzmax = Adcsc->nz + Bdcsc->nz;	// estimate on the size of resulting matrix C
    float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc->ConstructAux(nA, aux);
    
   
    
    
    // *************** Creating global space to store result, used by all threads *********************
    
    IT* maxnnzc = new IT[Bdcsc->nzc]; // maximum number of nnz in each column of C
    IT flops = 0; // total flops (multiplication) needed to generate C
#pragma omp parallel
    {
        IT tflops=0; //thread private flops
#pragma omp for
        for(int i=0; i < Bdcsc->nzc; ++i)
        {
            IT locmax = 0;
            IT nnzcol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
            //vector< pair<IT,IT> > colinds(nnzcol);
            //Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcol, colinds, aux, csize);
            bool found;
            IT* curptr = Bdcsc->ir + Bdcsc->cp[i];
            
            for(IT j = 0; j < nnzcol; ++j)
            {
                IT pos = Adcsc->AuxIndex(curptr[j], found, aux, csize);
                if(found)
                {
                    locmax = locmax + (Adcsc->cp[pos+1] - Adcsc->cp[pos]);
                }
                //locmax = locmax + (colinds[j].second - colinds[j].first);
                
            }
            
            maxnnzc[i] = locmax;
            tflops += locmax;
        }
#pragma omp critical
        {
            flops += tflops;
        }
    }
    
    
    int numThreads;
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    
    IT colPerThread [numThreads + 1]; // thread i will process columns from colPerThread[i] to colPerThread[i+1]-1
    colPerThread[0] = 0;
    
    
    IT* colStart = prefixsum<IT>(maxnnzc, Bdcsc->nzc, numThreads);
#pragma omp parallel for
    for(int i=1; i< numThreads; i++)
    {
        IT cur_col = i * (flops/numThreads);
        IT* it = std::lower_bound (colStart, colStart+Bdcsc->nzc+1, cur_col);
        colPerThread[i] = it - colStart;
        if(colPerThread[i]>Bdcsc->nzc) colPerThread[i]=Bdcsc->nzc;
    }
    colPerThread[numThreads] = Bdcsc->nzc;
    
   
    
    /*
    IT* colStart = new IT[Bdcsc->nzc]; //start index in the global array for storing ith column of C
    colStart[0] = 0;
    IT flopsPerThread = flops/numThreads; // amount of work that will be assigned to each thread
    int curThread = 1;
    IT nextflops = flopsPerThread;
    for(int i=0; i < (Bdcsc->nzc-1); ++i)
    {
        colStart[i+1] = colStart[i] + maxnnzc[i];
        if(nextflops < colStart[i+1])
        {
            colPerThread[curThread++] = i+1;
            nextflops += flopsPerThread;
        }
    }
    while(curThread < numThreads)
    colPerThread[curThread++] = Bdcsc->nzc;
    colPerThread[numThreads] = Bdcsc->nzc;
    */
    
    
    IT size = colStart[Bdcsc->nzc-1] + maxnnzc[Bdcsc->nzc-1];
    tuple<IT,IT,NTO> * tuplesC = static_cast<tuple<IT,IT,NTO> *> (::operator new (sizeof(tuple<IT,IT,NTO>[size])));
    
    delete [] maxnnzc;
    // ************************ End Creating global space *************************************
    
    // *************** Creating global heap space to be used by all threads *********************
    IT threadHeapSize[numThreads];
#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        IT localmax = -1;
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        {
            IT colnnz = Bdcsc->cp[i+1]-Bdcsc->cp[i];
            if(colnnz > localmax) localmax = colnnz;
        }
        threadHeapSize[thisThread] = localmax;
    }
    
    IT threadHeapStart[numThreads+1];
    threadHeapStart[0] = 0;
    for(int i=0; i<numThreads; i++)
    threadHeapStart[i+1] = threadHeapStart[i] + threadHeapSize[i];
    HeapEntry<IT,NT1> * globalheap = new HeapEntry<IT,NT1>[threadHeapStart[numThreads]];
    //HeapEntry<IT,NT1> * colinds1 = new HeapEntry<IT,NT1>[threadHeapStart[numThreads]];
    
    // ************************ End Creating global heap space *************************************
   
    //double t02 = MPI_Wtime();
    IT* colEnd = new IT[Bdcsc->nzc]; //end index in the global array for storing ith column of C

#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        vector< pair<IT,IT> > colinds(threadHeapSize[thisThread]);  //
        HeapEntry<IT,NT1> * wset = globalheap + threadHeapStart[thisThread]; // thread private heap space
        
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        {
            
            
            IT nnzcol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
            colEnd[i] = colStart[i];
            
            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
            //vector< pair<IT,IT> > colinds(nnzcol);
            Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcol, colinds, aux, csize); // can be done multithreaded
            IT hsize = 0;
            
            for(IT j = 0; (unsigned)j < nnzcol; ++j)		// create the initial heap
            {
                if(colinds[j].first != colinds[j].second)	// current != end
                {
                    wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc->ir[colinds[j].first], j, Adcsc->numx[colinds[j].first]);
                }
            }
            make_heap(wset, wset+hsize);
            
            
            while(hsize > 0)
            {
                pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
                IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column
                
                NTO mrhs = SR::multiply(wset[hsize-1].num, Bdcsc->numx[Bdcsc->cp[i]+locb]);
                if (!SR::returnedSAID())
                {
                    if( (colEnd[i] > colStart[i]) && get<0>(tuplesC[colEnd[i]-1]) == wset[hsize-1].key)
                    {
                        get<2>(tuplesC[colEnd[i]-1]) = SR::add(get<2>(tuplesC[colEnd[i]-1]), mrhs);
                    }
                    else
                    {
                        tuplesC[colEnd[i]]= make_tuple(wset[hsize-1].key, Bdcsc->jc[i], mrhs) ;
                        colEnd[i] ++;
                    }
                    
                }
                
                if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
                {
                    // runr stays the same !
                    wset[hsize-1].key = Adcsc->ir[colinds[locb].first];
                    wset[hsize-1].num = Adcsc->numx[colinds[locb].first];
                    push_heap(wset, wset+hsize);
                }
                else
                {
                    --hsize;
                }
            }
        }
        
    }
    
    //double t03 = MPI_Wtime();
    delete [] aux;
    delete [] globalheap;
    
    
    vector<IT> nnzcol(Bdcsc->nzc);
#pragma omp parallel for
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        nnzcol[i] = colEnd[i]-colStart[i];
    }
    
    IT* colptrC = prefixsum<IT>(nnzcol.data(), Bdcsc->nzc, numThreads); //parallel
    
    /*
    IT* colptrC = new IT[Bdcsc->nzc+1];
    colptrC[0] = 0;
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        colptrC[i+1] = colptrCt[i] +nnzcol[i];
    }
    
    */
    


    
    

    
    IT nnzc = colptrC[Bdcsc->nzc];
    tuple<IT,IT,NTO> * tuplesOut = static_cast<tuple<IT,IT,NTO> *> (::operator new (sizeof(tuple<IT,IT,NTO>[nnzc])));
    
    //double t05 = MPI_Wtime();
#pragma omp parallel for
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        copy(&tuplesC[colStart[i]], &tuplesC[colEnd[i]], tuplesOut + colptrC[i]);
    }

    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    delete [] tuplesC; // this consumes a significant amout of time on 12 cores e.g., .1s for scale=20
    delete [] colStart;
    delete [] colEnd;
    delete [] colptrC;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesOut, true);
    
    //cout << " last " << t06-t05 << " + " << t07-t06 << " + " << t08-t07 << " + " << " seconds" << endl;
    
    //cout << " local SpGEMM " << t02-t01 << " + " << t03-t02 << " + " << MPI_Wtime()-t03 << " seconds" << endl;
    
    return spTuplesC;
}


/***************************************************************************
 * Merging a list of tuples (multithreaded).
 * Two steps:   (a) Merge list by keeping duplicate entries
 *              (b) Reduce the merge list by adding duplicate entries
 * Inputs:
 *      listTuples: a vector of SpTuples to be merged
 *      deltuples: whether listTuples are deleted upon return
 *  Output:
 *      An object of SpTuples containing merged tuples.
 ***************************************************************************/

template<class IT, class NT>
SpTuples<IT,NT>*  multiwayMerge( const vector< SpTuples<IT,NT>* >& listTuples, bool deltuples = false )
{
    
    
    double t01 = MPI_Wtime();
    IT mdim = listTuples[0]->getnrow();
    IT ndim = listTuples[0]->getncol();
    
    IT totSize = 0;
    
    // ------- format input for __gnu_parallel::multiway_merge -------
    vector<pair<tuple<IT, IT, NT>*, tuple<IT, IT, NT>* > > seqs;
    for(int i = 0; i < listTuples.size(); ++i)
    {
        seqs.push_back(make_pair(listTuples[i]->tuples, listTuples[i]->tuples + listTuples[i]->getnnz()));
        totSize += listTuples[i]->getnnz();
    }
    
    // ------- merge lists with __gnu_parallel::multiway_merge -------
    ColLexiCompare<IT,NT> comp;
    tuple<IT, IT, NT>* mergedTuples = static_cast<tuple<IT, IT, NT>*> (::operator new (sizeof(tuple<IT, IT, NT>[totSize])));
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), mergedTuples, totSize , comp);
    
    
    if(deltuples)
    {
        for(size_t i=0; i<listTuples.size(); ++i)
            delete listTuples[i];
    }
    
    
    // -------------------------------------------------------------------------------------
    // Parallel reduction.
    // Each thread is given equal part of the merged (unreduced) list.
    // Additional reduction might be needed at the first/last entries processed by each thread
    // -------------------------------------------------------------------------------------
    
    t01 = MPI_Wtime();
    int totThreads;
#pragma omp parallel
    {
        totThreads = omp_get_num_threads();
    }
    
    vector <IT> tstart(totThreads); // start position for each thread
    vector <IT> tend(totThreads); // end position for each thread
    vector <IT> tdisp(totThreads+1);
    tuple<IT, IT, NT>* reducedTuples = static_cast<tuple<IT, IT, NT>*> (::operator new (sizeof(tuple<IT, IT, NT>[totSize]))); // separate memory used for better thread scaling
    //cout << totSize << " entries merged in " << MPI_Wtime()-t01 << " seconds" << endl;

#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        IT start = threadID * (totSize / totThreads);
        IT end = (threadID + 1) * (totSize / totThreads);
        if(threadID == (totThreads-1)) end = totSize;
        
        IT curpos = start;
        if(end>start) reducedTuples[curpos] = mergedTuples[curpos];
        
        for (IT i = start+1; i < end; ++i)
        {
            if((get<0>(mergedTuples[i]) == get<0>(reducedTuples[curpos])) && (get<1>(mergedTuples[i]) == get<1>(reducedTuples[curpos])))
            {
                get<2>(reducedTuples[curpos]) += get<2>(mergedTuples[i]);
            }
            else
            {
                reducedTuples[++curpos] = mergedTuples[i];
            }
        }
        tstart[threadID] = start;
        if(end>start) tend[threadID] = curpos+1;
        else tend[threadID] = end; // start=end
    }
    double t02 = MPI_Wtime();
    // Additional reduce at the first/last entries processed by each thread
    // serially performed
    for(int t=totThreads-1; t>0; --t)
    {
        if(tend[t] > tstart[t] && tend[t-1] > tstart[t-1])
        {
            if((get<0>(reducedTuples[tstart[t]]) == get<0>(reducedTuples[tend[t-1]-1])) && (get<1>(reducedTuples[tstart[t]]) == get<1>(reducedTuples[tend[t-1]-1])))
            {
                get<2>(reducedTuples[tend[t-1]-1]) += get<2>(reducedTuples[tstart[t]]);
                tstart[t] ++;
            }
        }
    }
    
    tdisp[0] = 0;
    for(int t=0; t<totThreads; ++t)
    {
        tdisp[t+1] = tdisp[t] + tend[t] - tstart[t];
    }
    
    // ------------- Remove gaps between tuples processed by threads ------------
    IT mergedListSize = tdisp[totThreads];
    tuple<IT, IT, NT>* shrunkTuples = static_cast<tuple<IT, IT, NT>*> (::operator new (sizeof(tuple<IT, IT, NT>[mergedListSize])));
    
#pragma omp parallel // canot be done in parallel on the same array
    {
        int threadID = omp_get_thread_num();
        std::copy(reducedTuples + tstart[threadID], reducedTuples + tend[threadID], shrunkTuples + tdisp[threadID]);
    }
    
    delete [] mergedTuples;
    delete [] reducedTuples;
    
    SpTuples<IT, NT>* mergedSpTuples = new SpTuples<IT, NT> (mergedListSize, mdim, ndim, shrunkTuples, true);
    return mergedSpTuples;
    
    //cout << mergedListSize << " entries reduced in " << t02-t01 << " + " << t03-t02 << " + " << MPI_Wtime()-t03 <<" seconds" << endl;
}




