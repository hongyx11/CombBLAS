#define DETERMINISTIC
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif
#endif

#ifdef _OPENMP
	#include <omp.h>
#endif


double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;
using namespace combblas;

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}

/**
 * Binary function to prune the previously discovered vertices from the current frontier 
 * When used with EWiseApply(SparseVec V, DenseVec W,...) we get the 'exclude = false' effect of EWiseMult
**/
struct prunediscovered: public std::binary_function<int64_t, int64_t, int64_t >
{
  	int64_t operator()(int64_t x, const int64_t & y) const
	{
		return ( y == -1 ) ? x: -1;
	}
};

static void MPI_randuniq(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
    RandReduce<int64_t> RR;
    int64_t * inveccast = (int64_t *) invec;
    int64_t * inoutveccast = (int64_t *) inoutvec;
    for (int i=0; i<*len; i++ )
        inoutveccast[i] = RR(inveccast[i], inoutveccast[i]);
}

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./scbfs <Scale>" << endl;
			cout << "Example: mpirun -np 4 ./scbfs 20" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int32_t> SR;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
		typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

		// Declare objects
		PSpMat_Bool A;	
		PSpMat_s32p64 Aeff;
		FullyDistVec<int64_t, int64_t> degrees;	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
		unsigned scale;
		OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits

		scale = static_cast<unsigned>(atoi(argv[1]));
		ostringstream outs;
        outs << "Forcing scale to : " << scale << endl;
		
        // this is an undirected graph, so A*x does indeed BFS
        double initiator[4] = {.57, .19, .19, .05};

        double t01 = MPI_Wtime();
        double t02;
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
		t02 = MPI_Wtime();
		ostringstream tinfo;
		tinfo << "Generation took " << t02-t01 << " seconds" << endl;
		SpParHelper::Print(tinfo.str());


        // Start Kernel #1
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // conversion from distributed edge list, keeps self-loops, sums duplicates
        PSpMat_s32p64_Int * G = new PSpMat_s32p64_Int(*DEL, false);
        delete DEL;	// free memory before symmetricizing
        SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");

        MPI_Barrier(MPI_COMM_WORLD);
        double redts = MPI_Wtime();
        G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0
        MPI_Barrier(MPI_COMM_WORLD);
        double redtf = MPI_Wtime();

        ostringstream redtimeinfo;
        redtimeinfo << "Calculated degrees in " << redtf-redts << " seconds" << endl;
        SpParHelper::Print(redtimeinfo.str());
        A =  PSpMat_Bool(*G);			// Convert to Boolean
        delete G;
        int64_t removed  = A.RemoveLoops();

        ostringstream loopinfo;
        loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
        SpParHelper::Print(loopinfo.str());
        A.PrintInfo();

        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
        FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
        A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
        A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
        SpParHelper::Print("Reductions done\n");
        ColSums->EWiseApply(*RowSums, plus<int64_t>());
        delete RowSums;
        SpParHelper::Print("Intersection of colsums and rowsums found\n");
		// only the indices of non-isolated vertices
        nonisov = ColSums->FindInds([](int64_t val){return val > 0;});
        delete ColSums;

        SpParHelper::Print("Found (and permuted) non-isolated vertices\n");
        nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
        A.PrintInfo();
    #ifndef NOPERMUTE
        A(nonisov, nonisov, true);	// in-place permute to save memory
        SpParHelper::Print("Dropped isolated vertices from input\n");
        A.PrintInfo();
    #endif
		
        Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
        A.FreeMemory();
        Symmetricize(Aeff);	// A += A';
        SpParHelper::Print("Symmetricized\n");

        Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
    #ifdef THREADED
        ostringstream tinfo;
        tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
        SpParHelper::Print(tinfo.str());
        Aeff.ActivateThreading(cblas_splits);
    #endif
        Aeff.PrintInfo();
			
        MPI_Barrier(MPI_COMM_WORLD);
        double t2=MPI_Wtime();
        
        ostringstream k1timeinfo;
        k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
        SpParHelper::Print(k1timeinfo.str());
		
		Aeff.PrintInfo();
		float balance = Aeff.LoadImbalance();
		ostringstream outs2;
		outs2 << "Load balance: " << balance << endl;
		SpParHelper::Print(outs2.str());

		MPI_Barrier(MPI_COMM_WORLD);

		// Now that every remaining vertex is non-isolated, randomly pick ITERS many of them as starting vertices
		#ifndef NOPERMUTE
		degrees = degrees(nonisov);	// fix the degrees array too
		degrees.PrintInfo("Degrees array");
		#endif
		// degrees.DebugPrint();
		FullyDistVec<int64_t, int64_t> Cands(ITERS, 0);
		double nver = (double) degrees.TotalLength();

#ifdef DETERMINISTIC
		MTRand M(1);
#else
		MTRand M;	// generate random numbers with Mersenne Twister 
#endif
		vector<double> loccands(ITERS);
		vector<int64_t> loccandints(ITERS);
		if(myrank == 0)
		{
			for(int i=0; i<ITERS; ++i)
				loccands[i] = M.rand();
			copy(loccands.begin(), loccands.end(), ostream_iterator<double>(cout," ")); cout << endl;
			transform(loccands.begin(), loccands.end(), loccands.begin(),
				[nver](double val){return val * nver;}
				);
			
			for(int i=0; i<ITERS; ++i)
				loccandints[i] = static_cast<int64_t>(loccands[i]);
			copy(loccandints.begin(), loccandints.end(), ostream_iterator<double>(cout," ")); cout << endl;
		}
		MPI_Bcast(&(loccandints[0]), ITERS, MPIType<int64_t>(),0,MPI_COMM_WORLD);
		for(int i=0; i<ITERS; ++i)
			Cands.SetElement(i,loccandints[i]);

        double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];
        for(int i=0; i<ITERS; ++i)
        {
            // FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
            FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1

            // FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
            FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();

            fringe.SetElement(Cands[i], Cands[i]);
            int iterations = 0;
            
            MPI_Op randreducempiop;
            MPI_Op_create(MPI_randuniq, true, &randreducempiop);
            
            while(fringe.getnnz() > 0)
            {
                fringe.setNumToInd();
                fringe = SpMV(Aeff, fringe,optbuf);	// SpMV with sparse vector (with indexisvalue flag preset), optimization enabled
                fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents
                fringe.PrintInfo("Frontier");
                auto singlechild = fringe.Uniq(RandReduce<int64_t>(), randreducempiop);
                singlechild.PrintInfo("Single child frontier");
                
                parents.Set(fringe);
                iterations++;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();
	
            FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find([](int64_t val){return val > -1;});
            parentsp.Apply(myset<int64_t>(1));
	
            // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
            int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
	
            ostringstream outnew;
            outnew << i << "th starting vertex was " << Cands[i] << endl;
            outnew << "Number iterations: " << iterations << endl;
            outnew << "Number of vertices found: " << parentsp.Reduce(plus<int64_t>(), (int64_t) 0) << endl;
            outnew << "Number of edges traversed: " << nedges << endl;
            outnew << "BFS time: " << t2-t1 << " seconds" << endl;
            outnew << "MTEPS: " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
            outnew << "Total communication (average so far): " << (cblas_allgathertime + cblas_alltoalltime) / (i+1) << endl;
            TIMES[i] = t2-t1;
            EDGES[i] = nedges;
            MTEPS[i] = static_cast<double>(nedges) / (t2-t1) / 1000000.0;
            SpParHelper::Print(outnew.str());
        }
        SpParHelper::Print("Finished\n");
        ostringstream os;
			

        sort(EDGES, EDGES+ITERS);
        os << "--------------------------" << endl;
        os << "Min nedges: " << EDGES[0] << endl;
        os << "First Quartile nedges: " << (EDGES[(ITERS/4)-1] + EDGES[ITERS/4])/2 << endl;
        os << "Median nedges: " << (EDGES[(ITERS/2)-1] + EDGES[ITERS/2])/2 << endl;
        os << "Third Quartile nedges: " << (EDGES[(3*ITERS/4) -1 ] + EDGES[3*ITERS/4])/2 << endl;
        os << "Max nedges: " << EDGES[ITERS-1] << endl;
        double mean = accumulate( EDGES, EDGES+ITERS, 0.0 )/ ITERS;
        vector<double> zero_mean(ITERS);	// find distances to the mean
        transform(EDGES, EDGES+ITERS, zero_mean.begin(),
        	[mean](double val){return val - mean;});
        // self inner-product is sum of sum of squares
        double deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
        deviation = sqrt( deviation / (ITERS-1) );
        os << "Mean nedges: " << mean << endl;
        os << "STDDEV nedges: " << deviation << endl;
        os << "--------------------------" << endl;
	
        sort(TIMES,TIMES+ITERS);
        os << "Min time: " << TIMES[0] << " seconds" << endl;
        os << "First Quartile time: " << (TIMES[(ITERS/4)-1] + TIMES[ITERS/4])/2 << " seconds" << endl;
        os << "Median time: " << (TIMES[(ITERS/2)-1] + TIMES[ITERS/2])/2 << " seconds" << endl;
        os << "Third Quartile time: " << (TIMES[(3*ITERS/4)-1] + TIMES[3*ITERS/4])/2 << " seconds" << endl;
        os << "Max time: " << TIMES[ITERS-1] << " seconds" << endl;
        mean = accumulate( TIMES, TIMES+ITERS, 0.0 )/ ITERS;
        transform(TIMES, TIMES+ITERS, zero_mean.begin(),
        	[mean](double val){return val - mean;});
        deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
        deviation = sqrt( deviation / (ITERS-1) );
        os << "Mean time: " << mean << " seconds" << endl;
        os << "STDDEV time: " << deviation << " seconds" << endl;
        os << "--------------------------" << endl;

        sort(MTEPS, MTEPS+ITERS);
        os << "Min MTEPS: " << MTEPS[0] << endl;
        os << "First Quartile MTEPS: " << (MTEPS[(ITERS/4)-1] + MTEPS[ITERS/4])/2 << endl;
        os << "Median MTEPS: " << (MTEPS[(ITERS/2)-1] + MTEPS[ITERS/2])/2 << endl;
        os << "Third Quartile MTEPS: " << (MTEPS[(3*ITERS/4)-1] + MTEPS[3*ITERS/4])/2 << endl;
        os << "Max MTEPS: " << MTEPS[ITERS-1] << endl;
        transform(MTEPS, MTEPS+ITERS, INVMTEPS, safemultinv<double>()); 	// returns inf for zero teps
        double hteps = static_cast<double>(ITERS) / accumulate(INVMTEPS, INVMTEPS+ITERS, 0.0);
        os << "Harmonic mean of MTEPS: " << hteps << endl;
        transform(INVMTEPS, INVMTEPS+ITERS, zero_mean.begin(),
        	[hteps](double val){return val - 1/hteps;});
        deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
        deviation = sqrt( deviation / (ITERS-1) ) * (hteps*hteps);	// harmonic_std_dev
        os << "Harmonic standard deviation of MTEPS: " << deviation << endl;
        SpParHelper::Print(os.str());
	}
	MPI_Finalize();
	return 0;
}

