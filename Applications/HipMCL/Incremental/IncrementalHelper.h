// functions used in Incremental Projects
#pragma once
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/SpParHelper.h"

namespace combblas {

 
/*
 * A^2 with incremental MCL matrix
 * Non-zeroes are heavily skewed on the diagonals, hence SUMMA is suboptimal
 * We seprate diagonal elements from offdiagonals, M = D + A; D is diagonal and A is off-diagonal matrix;
 * D can be thought of sparse vector, but we use dense vector here to avoid technical difficult;
 * M^2 = (D+A)^2 = D^2 + A^2 + DxA + AxD
 * A^2: SUMMA (Verify whether SUMMA or 1D multiplication would be optimal?)
 * D^2: Elementwise squaring of vector
 * DxA: Vector dimapply along row of A
 * AxD: Vector dimapply along column of A
 * */
template <typename SR, typename ITA, typename NTA, typename DERA>
SpParMat<ITA, NTA, DERA> IncrementalMCLSquare(SpParMat<ITA, NTA, DERA> & A,
                                           int phases, NTA hardThreshold, ITA selectNum, ITA recoverNum, NTA recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t0, t1, t2, t3, t4, t5;

    // Because we are squaring A, it is safe to assume that same commgrid as A would be used for all distributed matrices
    std::shared_ptr<CommGrid> commGrid = A.getcommgrid();
    MPI_Comm rowWorld = commGrid->GetRowWorld();
    MPI_Comm colWorld = commGrid->GetColWorld();

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t0 = MPI_Wtime();
#endif
    
    SpParMat<ITA, NTA, DERA> X(commGrid);
    {
        // Doing this inside here to force destruction of temporary objects once X is computed
        
        // MTH: What are mechanisms exist in CombBLAS to separate the diagonal elements?
        SpParMat<ITA, NTA, DERA> D(A);
        A.RemoveLoops(); // Remove diagonals, makes A as off-diagonal matrix
        D.SetDifference(A); // Remove offdiagonals

        FullyDistVec<ITA, NTA> diag = D.Reduce(Column, plus<NTA>(), 0.0); // diag: Vector with diagonal entries of D

        SpParMat<ITA, NTA, DERA> AD(A);
        AD.DimApply(Column, diag, [](NTA mv, NTA vv){return mv * vv;});
        AD.Prune(std::bind2nd(std::less_equal<NTA>(), 1e-8), true);

        SpParMat<ITA, NTA, DERA> DA(A);
        DA.DimApply(Row, diag, [](NTA mv, NTA vv){return mv * vv;});
        DA.Prune(std::bind2nd(std::less_equal<NTA>(), 1e-8), true);

        X = D;
        X.Apply(bind2nd(exponentiate(), 2));

        X += DA;
        X += AD;
    }
#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t1 = MPI_Wtime();
    if(myrank == 0){
        fprintf(stderr, "[IncrementalMCLSquare]:\tTime to calculate AD+DA+D^2: %lf\n", t1-t0);
    }
#endif

    if(phases < 1 || phases >= A.getncol())
    {
        SpParHelper::Print("[IncrementalMCLSquare]:\tThe value of phases is too small or large. Resetting to 1.\n");
        phases = 1;
    }
    
    int stages = commGrid->GetGridRows(); // As we use square grid number of rows would also mean number of columns in the grid
    float lb = A.LoadImbalance();
    //if(myrank == 0) fprintf(stderr, "[IncrementalMClSquare]:\tLoad imbalance of the matrix involved in SUMMA: %f\n", lb);

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t0 = MPI_Wtime();
#endif
    if(perProcessMemory>0) // estimate the number of phases permitted by memory
    {
        //int p;
        //MPI_Comm World = commGrid->GetWorld();
        //MPI_Comm_size(World,&p);
        
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.1\n");

        //int64_t perNNZMem_in = sizeof(ITA)*2 + sizeof(NTA);
        //int64_t perNNZMem_out = sizeof(ITA)*2 + sizeof(NTA);

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.2\n");
        
        //// max nnz(A) in a process
        //int64_t lannz = A.getlocalnnz();
        //int64_t gannz;
        //MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, World);
        //int64_t inputMem = gannz * perNNZMem_in * 5; // for five copies (two for SUMMA, one for X)
                                                    
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.3\n");
        
        //// max nnz(A^2) stored by SUMMA in a process
        //SpParMat<ITA, NTA, DERA> B(A);
        //int64_t asquareNNZ = EstPerProcessNnzSUMMA(A,B, false);
        //int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; // an extra copy in multiway merge and in selection/recovery step
                                                            
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.4\n");
        
        //// estimate kselect memory
        //int64_t d = ceil( (asquareNNZ * sqrt(p))/ A.getlocalcols() ); // average nnz per column in A^2 (it is an overestimate because asquareNNZ is estimated based on unmerged matrices)
        //// this is equivalent to (asquareNNZ * p) / A.getcol()
        //int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
        //int64_t kselectmem = A.getlocalcols() * k * 8 * 3;

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.5\n");
        
        //// estimate output memory
        //int64_t outputNNZ = (A.getlocalcols() * k)/sqrt(p);
        //int64_t outputMem = outputNNZ * perNNZMem_in * 2;

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.6\n");
        
        ////inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        //int64_t remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
        //if(remainingMem > 0)
        //{
            //phases = 1 + (asquareMem+kselectmem) / remainingMem;
        //}
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.7\n");
        
        
        //if(myrank==0)
        //{
            //if(remainingMem < 0)
            //{
                //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n Warning: input and output memory requirement is greater than per-process avaiable memory. Keeping phase to the value supplied at the command line. The program may go out of memory and crash! \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            //}
//#ifdef SHOW_MEMORY_USAGE
            //int64_t maxMemory = kselectmem/phases + inputMem + outputMem + asquareMem / phases;
            //if(maxMemory>1000000000)
            //std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000000.00 << " GB" << " inputMem: " << inputMem/1000000000.00 << " GB" << " outputMem: " << outputMem/1000000000.00 << " GB" << " kselectmem: " << kselectmem/1000000000.00 << " GB" << std::endl;
            //else
            //std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000.00 << " MB" << " inputMem: " << inputMem/1000000.00 << " MB" << " outputMem: " << outputMem/1000000.00 << " MB" << " kselectmem: " << kselectmem/1000000.00 << " MB" << std::endl;
//#endif
            
        //}
    }

    if(myrank == 0){
        fprintf(stderr, "[IncrementalMCLSquare]:\tRunning with phase: %d\n", phases);
    }

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t1 = MPI_Wtime();
    mcl_symbolictime += (t1-t0);
#endif
    
    ITA C_m = A.seqptr()->getnrow();
    ITA C_n = A.seqptr()->getncol();

    std::vector<DERA> PiecesOfB;
    DERA CopyA = *(A.seqptr()); // CopyA is effectively B because of A^2 computation
    CopyA.ColSplit(phases, PiecesOfB); // CopyA's memory is destroyed at this point
    
    std::vector<DERA> PiecesOfX;
    DERA CopyX = *(X.seqptr()); // Make a copy in order to use the ColSplit function
    CopyX.ColSplit(phases, PiecesOfX); // CopyX's memory is destroyed at this point

    X.FreeMemory(); // X is not needed anymore after splitting into `phases` pieces
    MPI_Barrier(commGrid->GetWorld());
    
    ITA ** ARecvSizes = SpHelper::allocate2D<ITA>(DERA::esscount, stages);
    ITA ** BRecvSizes = SpHelper::allocate2D<ITA>(DERA::esscount, stages);
    
    SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes, commGrid->GetRowWorld());
    
    // Remotely fetched matrices are stored as pointers
    DERA * ARecv;
    DERA * BRecv;
    
    std::vector< DERA > toconcatenate;
    
    int Aself = commGrid->GetRankInProcRow();
    int Bself = commGrid->GetRankInProcCol();

    stringstream strn;

    for(int p = 0; p< phases; ++p)
    {
        SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes, colWorld);
        std::vector< SpTuples<ITA,NTA>  *> tomerge;

        SpTuples<ITA,NTA> * PieceOfX = new SpTuples<ITA,NTA>(PiecesOfX[p]); // Convert target piece of X to SpTuples
        tomerge.push_back(PieceOfX); // Will be merged together with the result of A^2 with non-diagonal entries

        for(int i = 0; i < stages; ++i)
        {
            std::vector<ITA> ess;
            if(i == Aself)  ARecv = A.seqptr();	// shallow-copy
            else
            {
                ess.resize(DERA::esscount);
                for(int j=0; j< DERA::esscount; ++j)
                    ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
                ARecv = new DERA();				// first, create the object
            }
            
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            t0 = MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(commGrid->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            t1 = MPI_Wtime();
            mcl_Abcasttime += (t1-t0);
            /*
            int64_t nnz_local = ARecv->getnnz();
            int64_t nnz_min;
            int64_t nnz_max;
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_nnz_max: " << nnz_max << ", A_nnz_min: " << nnz_min << std::endl;;
            double time_local = t1-t0;
            double time_min;
            double time_max;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_bcast_time_max: " << time_max << ", A_bcast_time_min: " << time_min << std::endl;;
            */

#endif
            ess.clear();

            if(i == Bself)  BRecv = &(PiecesOfB[p]);	// shallow-copy
            else
            {
                ess.resize(DERA::esscount);
                for(int j=0; j< DERA::esscount; ++j)
                    ess[j] = BRecvSizes[j][i];
                BRecv = new DERA();
            }
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t2=MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(commGrid->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t3=MPI_Wtime();
            mcl_Bbcasttime += (t3-t2);
            /*
            nnz_local = BRecv->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_nnz_max: " << nnz_max << ", B_nnz_min: " << nnz_min << std::endl;;
            time_local = t3-t2;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_bcast_time_max: " << time_max << ", B_bcast_time_min: " << time_min << std::endl;;
            */
#endif
            
            
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t4=MPI_Wtime();
#endif
            SpTuples<ITA,NTA> * C_cont;
            //if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself, false); // Hash SpGEMM without per-column sorting
            //else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself);
            if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NTA>(*ARecv, *BRecv, false, false, false); // Hash SpGEMM without per-column sorting
            else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NTA>(*ARecv, *BRecv, false, false);
            
            // Explicitly delete ARecv and BRecv because it effectively does not get freed inside LocalSpGEMM function
            if(i != Bself && (!BRecv->isZero())) delete BRecv;
            if(i != Aself && (!ARecv->isZero())) delete ARecv;

#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t5=MPI_Wtime();
            mcl_localspgemmtime += (t5-t4);
            /*
            nnz_local = C_cont->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", C_nnz_max: " << nnz_max << ", C_nnz_min: " << nnz_min << std::endl;;
            time_local = t5-t4;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", spgemm_time_max: " << time_max << ", spgemm_time_min: " << time_min << std::endl;;
            */
#endif

            if(!C_cont->isZero())
                tomerge.push_back(C_cont);
            else
                delete C_cont;
            
        }   // all stages executed
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_unmerged, lcnnz_unmerged = 0;
        for(size_t i = 0; i < tomerge.size(); ++i)
        {
             lcnnz_unmerged += tomerge[i]->getnnz();
        }
        MPI_Allreduce(&lcnnz_unmerged, &gcnnz_unmerged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        int64_t summa_memory = gcnnz_unmerged*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gannz + gannz/phases) * 20; // last two for broadcasts
        
        if(myrank==0)
        {
            if(summa_memory>1000000000)
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000000.00 << "GB " ;
            else
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000.00 << " MB " ;
            
        }
#endif

#ifdef TIMING
        MPI_Barrier(commGrid->GetWorld());
        double t6=MPI_Wtime();
#endif
        // TODO: MultiwayMerge can directly return UDERO inorder to avoid the extra copy
        SpTuples<ITA,NTA> * OnePieceOfC_tuples;
        if(computationKernel == 1) OnePieceOfC_tuples = MultiwayMergeHash<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true, false);
        else if(computationKernel == 2) OnePieceOfC_tuples = MultiwayMerge<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true);
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_merged, lcnnz_merged ;
        lcnnz_merged = OnePieceOfC_tuples->getnnz();
        MPI_Allreduce(&lcnnz_merged, &gcnnz_merged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
       
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t merge_memory = gcnnz_merged*2*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gcnnz_merged*2) * 20;
        
        if(myrank==0)
        {
            if(merge_memory>1000000000)
                std::cout << " merged: " << merge_memory/1000000000.00 << "GB " ;
            else
                std::cout << " merged: " << merge_memory/1000000.00 << " MB " ;
        }
#endif
        
        
#ifdef TIMING
        MPI_Barrier(commGrid->GetWorld());
        double t7=MPI_Wtime();
        mcl_multiwaymergetime += (t7-t6);
#endif
        DERA * OnePieceOfC = new DERA(* OnePieceOfC_tuples, false);
        delete OnePieceOfC_tuples;
        
        SpParMat<ITA,NTA,DERA> OnePieceOfC_mat(OnePieceOfC, commGrid);
        MCLPruneRecoverySelect(OnePieceOfC_mat, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);

#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_pruned, lcnnz_pruned ;
        lcnnz_pruned = OnePieceOfC_mat.getlocalnnz();
        MPI_Allreduce(&lcnnz_pruned, &gcnnz_pruned, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        
        
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t prune_memory = gcnnz_pruned*2*20;//(gannz*2 + phase_nnz + gcnnz_pruned*2) * 20 + kselectmem; // 3 extra copies of OnePieceOfC_mat, we can make it one extra copy!
        //phase_nnz += gcnnz_pruned;
        
        if(myrank==0)
        {
            if(prune_memory>1000000000)
                std::cout << "Prune: " << prune_memory/1000000000.00 << "GB " << std::endl ;
            else
                std::cout << "Prune: " << prune_memory/1000000.00 << " MB " << std::endl ;
            
        }
#endif
        
        // ABAB: Change this to accept pointers to objects
        toconcatenate.push_back(OnePieceOfC_mat.seq());
    }
    SpParHelper::Print(strn.str());
    
    DERA * C = new DERA(0,C_m, C_n,0);
    C->ColConcatenate(toconcatenate); // ABAB: Change this to accept a vector of pointers to pointers to DERA objects

    SpHelper::deallocate2D(ARecvSizes, DERA::esscount);
    SpHelper::deallocate2D(BRecvSizes, DERA::esscount);
    return SpParMat<ITA,NTA,DERA> (C, commGrid);
}





}