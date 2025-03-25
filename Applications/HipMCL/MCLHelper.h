#pragma once
#include "CombBLAS/SpParMat.h"

extern double mcl_Abcasttime;
extern double mcl_Bbcasttime;
extern double mcl_localspgemmtime;
extern double mcl_multiwaymergetime;
extern double mcl_kselecttime;
extern double mcl_prunecolumntime;
extern double mcl_symbolictime;

extern double mcl3d_conversiontime;
extern double mcl3d_symbolictime;
extern double mcl3d_Abcasttime;
extern double mcl3d_Bbcasttime;
extern double mcl3d_SUMMAtime;
extern double mcl3d_localspgemmtime;
extern double mcl3d_SUMMAmergetime;
extern double mcl3d_reductiontime;
extern double mcl3d_3dmergetime;
extern double mcl3d_kselecttime;

namespace combblas {


/**
 * Broadcasts A multiple times (#phases) in order to save storage in the output
 * Only uses 1/phases of C memory if the threshold/max limits are proper
 * Parameters:
 *  - computationKernel: 1 means hash-based, 2 means heap-based
 */
 template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
 SpParMat<IU,NUO,UDERO> MemEfficientSpGEMM (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B,
                                            int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory)
 {
     typedef typename UDERA::LocalIT LIA;
     typedef typename UDERB::LocalIT LIB;
     typedef typename UDERO::LocalIT LIC;
     
     int myrank;
     MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
     if(A.getncol() != B.getnrow())
     {
         std::ostringstream outs;
         outs << "Can not multiply, dimensions does not match"<< std::endl;
         outs << A.getncol() << " != " << B.getnrow() << std::endl;
         SpParHelper::Print(outs.str());
         MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
         return SpParMat< IU,NUO,UDERO >();
     }
     if(phases <1 || phases >= A.getncol())
     {
         SpParHelper::Print("MemEfficientSpGEMM: The value of phases is too small or large. Resetting to 1.\n");
         phases = 1;
     }
     
     int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
     std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
     
     double t0, t1, t2, t3, t4, t5;
 #ifdef TIMING
     MPI_Barrier(A.getcommgrid()->GetWorld());
     t0 = MPI_Wtime();
 #endif
     if(perProcessMemory>0) // estimate the number of phases permitted by memory
     {
         int p;
         MPI_Comm World = GridC->GetWorld();
         MPI_Comm_size(World,&p);
         
         int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
         int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);
         
         // max nnz(A) in a porcess
         int64_t lannz = A.getlocalnnz();
         int64_t gannz;
         MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, World);
         int64_t inputMem = gannz * perNNZMem_in * 4; // for four copies (two for SUMMA)
         
         // max nnz(A^2) stored by SUMMA in a porcess
         int64_t asquareNNZ = EstPerProcessNnzSUMMA(A,B, false);
         int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; // an extra copy in multiway merge and in selection/recovery step
         
         
         // estimate kselect memory
         int64_t d = ceil( (asquareNNZ * sqrt(p))/ B.getlocalcols() ); // average nnz per column in A^2 (it is an overestimate because asquareNNZ is estimated based on unmerged matrices)
         // this is equivalent to (asquareNNZ * p) / B.getcol()
         int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
         int64_t kselectmem = B.getlocalcols() * k * 8 * 3;
         
         // estimate output memory
         int64_t outputNNZ = (B.getlocalcols() * k)/sqrt(p);
         int64_t outputMem = outputNNZ * perNNZMem_in * 2;
         
         //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
         int64_t remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
         if(remainingMem > 0)
         {
             phases = 1 + (asquareMem+kselectmem) / remainingMem;
         }
         
         
         if(myrank==0)
         {
             if(remainingMem < 0)
             {
                 std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n Warning: input and output memory requirement is greater than per-process avaiable memory. Keeping phase to the value supplied at the command line. The program may go out of memory and crash! \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
             }
 #ifdef SHOW_MEMORY_USAGE
             int64_t maxMemory = kselectmem/phases + inputMem + outputMem + asquareMem / phases;
             if(maxMemory>1000000000)
             std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000000.00 << " GB" << " inputMem: " << inputMem/1000000000.00 << " GB" << " outputMem: " << outputMem/1000000000.00 << " GB" << " kselectmem: " << kselectmem/1000000000.00 << " GB" << std::endl;
             else
             std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000.00 << " MB" << " inputMem: " << inputMem/1000000.00 << " MB" << " outputMem: " << outputMem/1000000.00 << " MB" << " kselectmem: " << kselectmem/1000000.00 << " MB" << std::endl;
 #endif
             
         }
     }
 
     //if(myrank == 0){
         //fprintf(stderr, "[MemEfficientSpGEMM] Running with phase: %d\n", phases);
     //}
 
 #ifdef TIMING
     MPI_Barrier(A.getcommgrid()->GetWorld());
     t1 = MPI_Wtime();
     mcl_symbolictime += (t1-t0);
 #endif
     
     LIA C_m = A.spSeq->getnrow();
     LIB C_n = B.spSeq->getncol();
     
     std::vector< UDERB > PiecesOfB;
     UDERB CopyB = *(B.spSeq); // we allow alias matrices as input because of this local copy
     
     CopyB.ColSplit(phases, PiecesOfB); // CopyB's memory is destroyed at this point
     MPI_Barrier(GridC->GetWorld());
     
     LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
     LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
     
     static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
     static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");
     
     
     SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
     
     // Remotely fetched matrices are stored as pointers
     UDERA * ARecv;
     UDERB * BRecv;
     
     std::vector< UDERO > toconcatenate;
     
     int Aself = (A.commGrid)->GetRankInProcRow();
     int Bself = (B.commGrid)->GetRankInProcCol();
 
     stringstream strn;
 
     for(int p = 0; p< phases; ++p)
     {
         SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes, (B.commGrid)->GetColWorld());
         std::vector< SpTuples<LIC,NUO>  *> tomerge;
         for(int i = 0; i < stages; ++i)
         {
             std::vector<LIA> ess;
             if(i == Aself)  ARecv = A.spSeq;	// shallow-copy
             else
             {
                 ess.resize(UDERA::esscount);
                 for(int j=0; j< UDERA::esscount; ++j)
                     ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
                 ARecv = new UDERA();				// first, create the object
             }
             
 #ifdef TIMING
             MPI_Barrier(A.getcommgrid()->GetWorld());
             t0 = MPI_Wtime();
 #endif
             SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
 #ifdef TIMING
             MPI_Barrier(A.getcommgrid()->GetWorld());
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
                 ess.resize(UDERB::esscount);
                 for(int j=0; j< UDERB::esscount; ++j)
                     ess[j] = BRecvSizes[j][i];
                 BRecv = new UDERB();
             }
 #ifdef TIMING
             MPI_Barrier(A.getcommgrid()->GetWorld());
             double t2=MPI_Wtime();
 #endif
             SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
 #ifdef TIMING
             MPI_Barrier(A.getcommgrid()->GetWorld());
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
             MPI_Barrier(A.getcommgrid()->GetWorld());
             double t4=MPI_Wtime();
 #endif
             SpTuples<LIC,NUO> * C_cont;
             //if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself, false); // Hash SpGEMM without per-column sorting
             //else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself);
             if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv, false, false, false); // Hash SpGEMM without per-column sorting
             else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv, false, false);
             
             // Explicitly delete ARecv and BRecv because it effectively does not get freed inside LocalSpGEMM function
             if(i != Bself && (!BRecv->isZero())) delete BRecv;
             if(i != Aself && (!ARecv->isZero())) delete ARecv;
 
 #ifdef TIMING
             MPI_Barrier(A.getcommgrid()->GetWorld());
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
         MPI_Barrier(A.getcommgrid()->GetWorld());
         double t6=MPI_Wtime();
 #endif
         // TODO: MultiwayMerge can directly return UDERO inorder to avoid the extra copy
         SpTuples<LIC,NUO> * OnePieceOfC_tuples;
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
         MPI_Barrier(A.getcommgrid()->GetWorld());
         double t7=MPI_Wtime();
         mcl_multiwaymergetime += (t7-t6);
 #endif
         UDERO * OnePieceOfC = new UDERO(* OnePieceOfC_tuples, false);
         delete OnePieceOfC_tuples;
         
         SpParMat<IU,NUO,UDERO> OnePieceOfC_mat(OnePieceOfC, GridC);
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
     
     UDERO * C = new UDERO(0,C_m, C_n,0);
     C->ColConcatenate(toconcatenate); // ABAB: Change this to accept a vector of pointers to pointers to DER objects
 
     SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
     SpHelper::deallocate2D(BRecvSizes, UDERA::esscount);
     return SpParMat<IU,NUO,UDERO> (C, GridC);
 }

 


// Combined logic for prune, recovery, and select
template <typename IT, typename NT, typename DER>
void MCLPruneRecoverySelect(SpParMat<IT,NT,DER> & A, NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct, int kselectVersion)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
#ifdef TIMING
    double t0, t1;
#endif
    
    // Prune and create a new pruned matrix
    SpParMat<IT,NT,DER> PrunedA = A.Prune(std::bind2nd(std::less_equal<NT>(), hardThreshold), false);
    // column-wise statistics of the pruned matrix
    FullyDistVec<IT,NT> colSums = PrunedA.Reduce(Column, std::plus<NT>(), 0.0);
    FullyDistVec<IT,NT> nnzPerColumnUnpruned = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    FullyDistVec<IT,NT> nnzPerColumn = PrunedA.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    //FullyDistVec<IT,NT> pruneCols(A.getcommgrid(), A.getncol(), hardThreshold);
    FullyDistVec<IT,NT> pruneCols(nnzPerColumn);
    pruneCols = hardThreshold;

    PrunedA.FreeMemory();

    FullyDistSpVec<IT,NT> recoverCols(nnzPerColumn, std::bind2nd(std::less<NT>(), recoverNum));
    
    // recover only when nnzs in unprunned columns are greater than nnzs in pruned column
    recoverCols = EWiseApply<NT>(recoverCols, nnzPerColumnUnpruned,
                                 [](NT spval, NT dval){return spval;},
                                 [](NT spval, NT dval){return dval > spval;},
                                 false, NT());

    
    recoverCols = recoverPct;
    // columns with nnz < r AND sum < recoverPct (pct)
    recoverCols = EWiseApply<NT>(recoverCols, colSums,
                                 [](NT spval, NT dval){return spval;},
                                 [](NT spval, NT dval){return dval < spval;},
                                 false, NT());

    IT nrecover = recoverCols.getnnz();

    if(nrecover > 0)
    {
#ifdef TIMING
        t0=MPI_Wtime();
#endif
        A.Kselect(recoverCols, recoverNum, kselectVersion);

#ifdef TIMING
        t1=MPI_Wtime();
        mcl_kselecttime += (t1-t0);
#endif

        pruneCols.Set(recoverCols);

#ifdef COMBBLAS_DEBUG
        std::ostringstream outs;
        outs << "Number of columns needing recovery: " << nrecover << std::endl;
        SpParHelper::Print(outs.str());
#endif
        
    }

    if(selectNum>0)
    {
        // remaining columns will be up for selection
        FullyDistSpVec<IT,NT> selectCols = EWiseApply<NT>(recoverCols, colSums,
                                                          [](NT spval, NT dval){return spval;},
                                                          [](NT spval, NT dval){return spval==-1;},
                                                          true, static_cast<NT>(-1));
        
        selectCols = selectNum;
        selectCols = EWiseApply<NT>(selectCols, nnzPerColumn,
                                    [](NT spval, NT dval){return spval;},
                                    [](NT spval, NT dval){return dval > spval;},
                                    false, NT());
        IT nselect = selectCols.getnnz();
        
        if(nselect > 0 )
        {
#ifdef TIMING
            t0=MPI_Wtime();
#endif
            A.Kselect(selectCols, selectNum, kselectVersion); // PrunedA would also work
#ifdef TIMING
            t1=MPI_Wtime();
            mcl_kselecttime += (t1-t0);
#endif
        
            pruneCols.Set(selectCols);
#ifdef COMBBLAS_DEBUG
            std::ostringstream outs;
            outs << "Number of columns needing selection: " << nselect << std::endl;
            SpParHelper::Print(outs.str());
#endif
#ifdef TIMING
            t0=MPI_Wtime();
#endif
            SpParMat<IT,NT,DER> selectedA = A.PruneColumn(pruneCols, std::less<NT>(), false);
#ifdef TIMING
            t1=MPI_Wtime();
            mcl_prunecolumntime += (t1-t0);
#endif
            if(recoverNum>0 ) // recovery can be attempted after selection
            {

                FullyDistVec<IT,NT> nnzPerColumn1 = selectedA.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
                FullyDistVec<IT,NT> colSums1 = selectedA.Reduce(Column, std::plus<NT>(), 0.0);
                selectedA.FreeMemory();
  
                // slected columns with nnz < recoverNum (r)
                selectCols = recoverNum;
                selectCols = EWiseApply<NT>(selectCols, nnzPerColumn1,
                                            [](NT spval, NT dval){return spval;},
                                            [](NT spval, NT dval){return dval < spval;},
                                            false, NT());
                
                // selected columns with sum < recoverPct (pct)
                selectCols = recoverPct;
                selectCols = EWiseApply<NT>(selectCols, colSums1,
                                            [](NT spval, NT dval){return spval;},
                                            [](NT spval, NT dval){return dval < spval;},
                                            false, NT());
                
                IT n_recovery_after_select = selectCols.getnnz();
                if(n_recovery_after_select>0)
                {
                    // mclExpandVector2 does it on the original vector
                    // mclExpandVector1 does it one pruned vector
#ifdef TIMING
                    t0=MPI_Wtime();
#endif
                    A.Kselect(selectCols, recoverNum, kselectVersion); // Kselect on PrunedA might give different result
#ifdef TIMING
                    t1=MPI_Wtime();
                    mcl_kselecttime += (t1-t0);
#endif
                    pruneCols.Set(selectCols);
                    
#ifdef COMBBLAS_DEBUG
                    std::ostringstream outs1;
                    outs1 << "Number of columns needing recovery after selection: " << nselect << std::endl;
                    SpParHelper::Print(outs1.str());
#endif
                }
                
            }
        }
    }

    // final prune
#ifdef TIMING
    t0=MPI_Wtime();
#endif
    A.PruneColumn(pruneCols, std::less<NT>(), true);
#ifdef TIMING
    t1=MPI_Wtime();
    mcl_prunecolumntime += (t1-t0);
#endif
    // Add loops for empty columns
    if(recoverNum<=0 ) // if recoverNum>0, recovery would have added nonzeros in empty columns
    {
        FullyDistVec<IT,NT> nnzPerColumnA = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
        FullyDistSpVec<IT,NT> emptyColumns(nnzPerColumnA, std::bind2nd(std::equal_to<NT>(), 0.0));
        emptyColumns = 1.00;
        //Ariful: We need a selective AddLoops function with a sparse vector
        //A.AddLoops(emptyColumns);
    }

}


/*
 * Parameters:
 *  - computationKernel: 1 for hash-based, 2 for heap-based
 * */
 template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
 SpParMat3D<IU, NUO, UDERO> MemEfficientSpGEMM3D(SpParMat3D<IU, NU1, UDERA> & A, SpParMat3D<IU, NU2, UDERB> & B,
            int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory){
     int myrank;
     MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
     typedef typename UDERA::LocalIT LIA;
     typedef typename UDERB::LocalIT LIB;
     typedef typename UDERO::LocalIT LIC;
 
     /* 
      * Check if A and B are multipliable 
      * */
     if(A.getncol() != B.getnrow()){
         std::ostringstream outs;
         outs << "Can not multiply, dimensions does not match"<< std::endl;
         outs << A.getncol() << " != " << B.getnrow() << std::endl;
         SpParHelper::Print(outs.str());
         MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
     }
 
     /* 
      * If provided number of phase is too low or too high then reset value of phase as 1 
      * */
     if(phases < 1 || phases >= B.getncol()){
         SpParHelper::Print("[MemEfficientSpGEMM3D]\tThe value of phases is too small or large. Resetting to 1.\n");
         phases = 1;
     }
     double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9; // To time different parts of the function
 #ifdef TIMING
     MPI_Barrier(B.getcommgrid3D()->GetWorld());
     t0 = MPI_Wtime();
 #endif
     /* 
      * If per process memory is provided then calculate number of phases 
      * Otherwise, proceed to multiplication.
      * */
     if(perProcessMemory > 0) {
         int p, calculatedPhases;
         MPI_Comm_size(A.getcommgrid3D()->GetLayerWorld(),&p);
         int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
         int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);
 
         int64_t lannz = A.GetLayerMat()->getlocalnnz();
         int64_t gannz = 0;
         // Get maximum number of nnz owned by one process
         MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, A.getcommgrid3D()->GetWorld()); 
         //int64_t ginputMem = gannz * perNNZMem_in * 4; // Four pieces per process: one piece of own A and B, one piece of received A and B
         int64_t ginputMem = gannz * perNNZMem_in * 5; // One extra copy for safety
         
         // Estimate per layer nnz after multiplication. After this estimation each process would know an estimation of
         // how many nnz the corresponding layer will have after the layerwise operation.
         int64_t asquareNNZ = EstPerProcessNnzSUMMA(*(A.GetLayerMat()), *(B.GetLayerMat()), true);
         int64_t gasquareNNZ;
         MPI_Allreduce(&asquareNNZ, &gasquareNNZ, 1, MPIType<int64_t>(), MPI_MAX, A.getcommgrid3D()->GetFiberWorld());
 
         // Atmost two copies, one of a process's own, another received from fiber reduction
         int64_t gasquareMem = gasquareNNZ * perNNZMem_out * 2; 
         // Calculate estimated average degree after multiplication
         int64_t d = ceil( ( ( gasquareNNZ / B.getcommgrid3D()->GetGridLayers() ) * sqrt(p) ) / B.GetLayerMat()->getlocalcols() );
         // Calculate per column nnz how left after k-select. Minimum of average degree and k-select parameters.
         int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
 
         //estimate output memory
         int64_t postKselectOutputNNZ = ceil(( (B.GetLayerMat()->getlocalcols() / B.getcommgrid3D()->GetGridLayers() ) * k)/sqrt(p)); // If kselect is run
         int64_t postKselectOutputMem = postKselectOutputNNZ * perNNZMem_out * 2;
         double remainingMem = perProcessMemory*1000000000 - ginputMem - postKselectOutputMem;
         int64_t kselectMem = B.GetLayerMat()->getlocalcols() * k * sizeof(NUO) * 3;
 
         //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
         if(remainingMem > 0){
             calculatedPhases = ceil( (gasquareMem + kselectMem) / remainingMem ); // If kselect is run
         }
         else calculatedPhases = -1;
 
         int gCalculatedPhases;
         MPI_Allreduce(&calculatedPhases, &gCalculatedPhases, 1, MPI_INT, MPI_MAX, A.getcommgrid3D()->GetFiberWorld());
         if(gCalculatedPhases > phases) phases = gCalculatedPhases;
     }
     else{
         // Do nothing
     }
 #ifdef TIMING
     MPI_Barrier(B.getcommgrid3D()->GetWorld());
     t1 = MPI_Wtime();
     mcl3d_symbolictime+=(t1-t0);
     //if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tSymbolic stage time: %lf\n", (t1-t0));
 #endif
         
         
     /*
      * Calculate, accross fibers, which process should get how many columns after redistribution
      * */
     vector<LIB> divisions3d;
     // Calculate split boundaries as if all contents of the layer is being re-distributed along fiber
     // These boundaries will be used later on
     B.CalculateColSplitDistributionOfLayer(divisions3d); 
 
     /*
      * Split B according to calculated number of phases
      * For better load balancing split B into nlayers*phases chunks
      * */
     vector<UDERB*> PiecesOfB;
     vector<UDERB*> tempPiecesOfB;
     UDERB CopyB = *(B.GetLayerMat()->seqptr());
     CopyB.ColSplit(divisions3d, tempPiecesOfB); // Split B into `nlayers` chunks at first
     for(int i = 0; i < tempPiecesOfB.size(); i++){
         vector<UDERB*> temp;
         tempPiecesOfB[i]->ColSplit(phases, temp); // Split each chunk of B into `phases` chunks
         for(int j = 0; j < temp.size(); j++){
             PiecesOfB.push_back(temp[j]);
         }
     }
 
     vector<UDERO> toconcatenate;
     //if(myrank == 0){
         //fprintf(stderr, "[MemEfficientSpGEMM3D]\tRunning with phase: %d\n", phases);
     //}
 
     for(int p = 0; p < phases; p++){
         /*
          * At the start of each phase take appropriate pieces from previously created pieces of local B matrix
          * Appropriate means correct pieces so that 3D-merge can be properly load balanced.
          * */
         vector<LIB> lbDivisions3d; // load balance friendly division
         LIB totalLocalColumnInvolved = 0;
         vector<UDERB*> targetPiecesOfB; // Pieces of B involved in current phase
         for(int i = 0; i < PiecesOfB.size(); i++){
             if(i % phases == p){
                 targetPiecesOfB.push_back(new UDERB(*(PiecesOfB[i])));
                 lbDivisions3d.push_back(PiecesOfB[i]->getncol());
                 totalLocalColumnInvolved += PiecesOfB[i]->getncol();
             }
         }
 
         /*
          * Create new local matrix by concatenating appropriately picked pieces
          * */
         UDERB * OnePieceOfB = new UDERB(0, (B.GetLayerMat())->seqptr()->getnrow(), totalLocalColumnInvolved, 0);
         OnePieceOfB->ColConcatenate(targetPiecesOfB);
         vector<UDERB*>().swap(targetPiecesOfB);
 
         /*
          * Create a new layer-wise distributed matrix with the newly created local matrix for this phase
          * This matrix is used in SUMMA multiplication of respective layer
          * */
         SpParMat<IU, NU2, UDERB> OnePieceOfBLayer(OnePieceOfB, A.getcommgrid3D()->GetLayerWorld());
 #ifdef TIMING
         t0 = MPI_Wtime();
 #endif
         /*
          *  SUMMA Starts
          * */
 
         int stages, dummy; 	// last two parameters of ProductGrid are ignored for this multiplication
         std::shared_ptr<CommGrid> GridC = ProductGrid((A.GetLayerMat()->getcommgrid()).get(), 
                                                       (OnePieceOfBLayer.getcommgrid()).get(), 
                                                       stages, dummy, dummy);		
         LIA C_m = A.GetLayerMat()->seqptr()->getnrow();
         LIB C_n = OnePieceOfBLayer.seqptr()->getncol();
 
         LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
         LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
         
         SpParHelper::GetSetSizes( *(A.GetLayerMat()->seqptr()), ARecvSizes, (A.GetLayerMat()->getcommgrid())->GetRowWorld() );
         SpParHelper::GetSetSizes( *(OnePieceOfBLayer.seqptr()), BRecvSizes, (OnePieceOfBLayer.getcommgrid())->GetColWorld() );
 
         // Remotely fetched matrices are stored as pointers
         UDERA * ARecv; 
         UDERB * BRecv;
         std::vector< SpTuples<LIC,NUO>  *> tomerge;
 
         int Aself = (A.GetLayerMat()->getcommgrid())->GetRankInProcRow();
         int Bself = (OnePieceOfBLayer.getcommgrid())->GetRankInProcCol();	
 
         double Abcast_time = 0;
         double Bbcast_time = 0;
         double Local_multiplication_time = 0;
         
         for(int i = 0; i < stages; ++i) {
             std::vector<LIA> ess;	
 
             if(i == Aself){
                 ARecv = A.GetLayerMat()->seqptr();	// shallow-copy 
             }
             else{
                 ess.resize(UDERA::esscount);
                 for(int j=0; j<UDERA::esscount; ++j) {
                     ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
                 }
                 ARecv = new UDERA();				// first, create the object
             }
 #ifdef TIMING
             t2 = MPI_Wtime();
 #endif
             if (Aself != i) {
                 ARecv->Create(ess);
             }
 
             Arr<LIA,NU1> Aarrinfo = ARecv->GetArrays();
 
             for(unsigned int idx = 0; idx < Aarrinfo.indarrs.size(); ++idx) {
                 MPI_Bcast(Aarrinfo.indarrs[idx].addr, Aarrinfo.indarrs[idx].count, MPIType<IU>(), i, GridC->GetRowWorld());
             }
 
             for(unsigned int idx = 0; idx < Aarrinfo.numarrs.size(); ++idx) {
                 MPI_Bcast(Aarrinfo.numarrs[idx].addr, Aarrinfo.numarrs[idx].count, MPIType<NU1>(), i, GridC->GetRowWorld());
             }
 #ifdef TIMING
             t3 = MPI_Wtime();
             mcl3d_Abcasttime += (t3-t2);
             Abcast_time += (t3-t2);
 #endif
             ess.clear();	
             if(i == Bself){
                 BRecv = OnePieceOfBLayer.seqptr();	// shallow-copy
             }
             else{
                 ess.resize(UDERB::esscount);		
                 for(int j=0; j<UDERB::esscount; ++j)	{
                     ess[j] = BRecvSizes[j][i];	
                 }	
                 BRecv = new UDERB();
             }
 
             MPI_Barrier(A.GetLayerMat()->getcommgrid()->GetWorld());
 #ifdef TIMING
             t2 = MPI_Wtime();
 #endif
             if (Bself != i) {
                 BRecv->Create(ess);	
             }
             Arr<LIB,NU2> Barrinfo = BRecv->GetArrays();
 
             for(unsigned int idx = 0; idx < Barrinfo.indarrs.size(); ++idx) {
                 MPI_Bcast(Barrinfo.indarrs[idx].addr, Barrinfo.indarrs[idx].count, MPIType<IU>(), i, GridC->GetColWorld());
             }
             for(unsigned int idx = 0; idx < Barrinfo.numarrs.size(); ++idx) {
                 MPI_Bcast(Barrinfo.numarrs[idx].addr, Barrinfo.numarrs[idx].count, MPIType<NU2>(), i, GridC->GetColWorld());
             }
 #ifdef TIMING
             t3 = MPI_Wtime();
             mcl3d_Bbcasttime += (t3-t2);
             Bbcast_time += (t3-t2);
 #endif
 
 #ifdef TIMING
             t2 = MPI_Wtime();
 #endif
             SpTuples<LIC,NUO> * C_cont;
             
             if(computationKernel == 1){
                 C_cont = LocalSpGEMMHash<SR, NUO>
                                     (*ARecv, *BRecv,    // parameters themselves
                                     false,         // 'delete A' condition
                                     false,         // 'delete B' condition
                                     false);             // not to sort each column
             }
             else if(computationKernel == 2){
                 C_cont = LocalSpGEMM<SR, NUO>
                                     (*ARecv, *BRecv,    // parameters themselves
                                     false,         // 'delete A' condition
                                     false);        // 'delete B' condition
             
             }
             if(i != Bself && (!BRecv->isZero())) delete BRecv;
             if(i != Aself && (!ARecv->isZero())) delete ARecv;
             
 #ifdef TIMING
             t3 = MPI_Wtime();
             mcl3d_localspgemmtime += (t3-t2);
             Local_multiplication_time += (t3-t2);
 #endif
             
             if(!C_cont->isZero()) tomerge.push_back(C_cont);
         }
 
         SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
         SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
 
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         SpTuples<LIC,NUO> * C_tuples;
         if(computationKernel == 1) C_tuples = MultiwayMergeHash<SR>(tomerge, C_m, C_n, true, true); // Delete input arrays and sort
         else if(computationKernel == 2) C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n, true); // Delete input arrays and sort
         
 #ifdef TIMING
         t3 = MPI_Wtime();
         mcl3d_SUMMAmergetime += (t3-t2);
 #endif
 
 #ifdef TIMING 
         if(myrank == 0){
             fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAbcast_time: %lf\n", p, Abcast_time);
             fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tBbcast_time: %lf\n", p, Bbcast_time);
             fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tLocal_multiplication_time: %lf\n", p, Local_multiplication_time);
             fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA Merge time: %lf\n", p, (t3-t2));
         }
 #endif
         /*
          *  SUMMA Ends
          * */
 #ifdef TIMING
         t1 = MPI_Wtime();
         mcl3d_SUMMAtime += (t1-t0);
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA time: %lf\n", p, (t1-t0));
 #endif
 
         /*
          * 3d-reduction starts
          * */
 #ifdef TIMING
         t0 = MPI_Wtime();
         t2 = MPI_Wtime();
 #endif
         MPI_Datatype MPI_tuple;
         MPI_Type_contiguous(sizeof(std::tuple<LIC,LIC,NUO>), MPI_CHAR, &MPI_tuple);
         MPI_Type_commit(&MPI_tuple);
         
         /*
          *  Create a profile with information regarding data to be sent and received between layers 
          *  These memory allocation needs to be `int` specifically because some of these arrays would be used in communication
          *  This is requirement is for MPI as MPI_Alltoallv takes pointer to integer exclusively as count and displacement
          * */
         int * sendcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
         int * sendprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
         int * sdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();
         int * recvcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
         int * recvprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
         int * rdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();
 
         vector<LIC> lbDivisions3dPrefixSum(lbDivisions3d.size());
         lbDivisions3dPrefixSum[0] = 0;
         std::partial_sum(lbDivisions3d.begin(), lbDivisions3d.end()-1, lbDivisions3dPrefixSum.begin()+1);
         ColLexiCompare<LIC,NUO> comp;
         LIC totsend = C_tuples->getnnz();
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAllocation of alltoall information: %lf\n", p, (t3-t2));
 #endif
         
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
 #pragma omp parallel for
         for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
             LIC start_col = lbDivisions3dPrefixSum[i];
             LIC end_col = lbDivisions3dPrefixSum[i] + lbDivisions3d[i];
             std::tuple<LIC, LIC, NUO> search_tuple_start(0, start_col, NUO());
             std::tuple<LIC, LIC, NUO> search_tuple_end(0, end_col, NUO());
             std::tuple<LIC, LIC, NUO>* start_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_start, comp);
             std::tuple<LIC, LIC, NUO>* end_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_end, comp);
             // This type casting is important from semantic point of view
             sendcnt[i] = (int)(end_it - start_it);
             sendprfl[i*3+0] = (int)(sendcnt[i]); // Number of nonzeros in ith chunk
             sendprfl[i*3+1] = (int)(A.GetLayerMat()->seqptr()->getnrow()); // Number of rows in ith chunk
             sendprfl[i*3+2] = (int)(lbDivisions3d[i]); // Number of columns in ith chunk
         }
         std::partial_sum(sendcnt, sendcnt+A.getcommgrid3D()->GetGridLayers()-1, sdispls+1);
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoall data ready: %lf\n", p, (t3-t2));
 #endif
 
         // Send profile ready. Now need to update the tuples to reflect correct column id after column split.
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
 #pragma omp parallel for schedule(static)
             for(int j = 0; j < sendcnt[i]; j++){
                 std::get<1>(C_tuples->tuples[sdispls[i]+j]) = std::get<1>(C_tuples->tuples[sdispls[i]+j]) - lbDivisions3dPrefixSum[i];
             }
         }
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoallv data ready: %lf\n", p, (t3-t2));
 #endif
 
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, A.getcommgrid3D()->GetFiberWorld());
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoall: %lf\n", p, (t3-t2));
 #endif
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         for(int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++) recvcnt[i] = recvprfl[i*3];
         std::partial_sum(recvcnt, recvcnt+A.getcommgrid3D()->GetGridLayers()-1, rdispls+1);
         LIC totrecv = std::accumulate(recvcnt,recvcnt+A.getcommgrid3D()->GetGridLayers(), static_cast<IU>(0));
         std::tuple<LIC,LIC,NUO>* recvTuples = static_cast<std::tuple<LIC,LIC,NUO>*> (::operator new (sizeof(std::tuple<LIC,LIC,NUO>[totrecv])));
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAllocation of receive data: %lf\n", p, (t3-t2));
 #endif
 
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         MPI_Alltoallv(C_tuples->tuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, A.getcommgrid3D()->GetFiberWorld());
         delete C_tuples;
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoallv: %lf\n", p, (t3-t2));
 #endif
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         vector<SpTuples<LIC, NUO>*> recvChunks(A.getcommgrid3D()->GetGridLayers());
 #pragma omp parallel for
         for (int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++){
             recvChunks[i] = new SpTuples<LIC, NUO>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], recvTuples + rdispls[i], true, false);
         }
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\trecvChunks creation: %lf\n", p, (t3-t2));
 #endif
 
 #ifdef TIMING
         t2 = MPI_Wtime();
 #endif
         // Free all memory except tempTuples; Because that is holding data of newly created local matrices after receiving.
         DeleteAll(sendcnt, sendprfl, sdispls);
         DeleteAll(recvcnt, recvprfl, rdispls); 
         MPI_Type_free(&MPI_tuple);
 #ifdef TIMING
         t3 = MPI_Wtime();
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMemory freeing: %lf\n", p, (t3-t2));
 #endif
         /*
          * 3d-reduction ends 
          * */
         
 #ifdef TIMING
         t1 = MPI_Wtime();
         mcl3d_reductiontime += (t1-t0);
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tReduction time: %lf\n", p, (t1-t0));
 #endif
 #ifdef TIMING
         t0 = MPI_Wtime();
 #endif
         /*
          * 3d-merge starts 
          * */
         SpTuples<LIC, NUO> * merged_tuples;
 
         if(computationKernel == 1) merged_tuples = MultiwayMergeHash<SR, LIC, NUO>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false, false); // Do not delete
         else if(computationKernel == 2) merged_tuples = MultiwayMerge<SR, LIC, NUO>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false); // Do not delete
 #ifdef TIMING
         t1 = MPI_Wtime();
         mcl3d_3dmergetime += (t1-t0);
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\t3D Merge time: %lf\n", p, (t1-t0));
 #endif
         /*
          * 3d-merge ends
          * */
 #ifdef TIMING
         t0 = MPI_Wtime();
 #endif
         // Do not delete elements of recvChunks, because that would give segmentation fault due to double free
         ::operator delete(recvTuples);
         for(int i = 0; i < recvChunks.size(); i++){
             recvChunks[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
             delete recvChunks[i]; // As the patch is used, now delete each element of recvChunks
         }
         vector<SpTuples<LIC,NUO>*>().swap(recvChunks); // As the patch is used, now delete recvChunks
 
         // This operation is not needed if result can be used and discareded right away
         // This operation is being done because it is needed by MCLPruneRecoverySelect
         UDERO * phaseResultant = new UDERO(*merged_tuples, false);
         delete merged_tuples;
         SpParMat<IU, NUO, UDERO> phaseResultantLayer(phaseResultant, A.getcommgrid3D()->GetLayerWorld());
         MCLPruneRecoverySelect(phaseResultantLayer, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);
 #ifdef TIMING
         t1 = MPI_Wtime();
         mcl3d_kselecttime += (t1-t0);
         if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMCLPruneRecoverySelect time: %lf\n",p, (t1-t0));
 #endif
         toconcatenate.push_back(phaseResultantLayer.seq());
 #ifdef TIMING
         if(myrank == 0) fprintf(stderr, "***\n");
 #endif
     }
     for(int i = 0; i < PiecesOfB.size(); i++) delete PiecesOfB[i];
 
     std::shared_ptr<CommGrid3D> grid3d;
     grid3d.reset(new CommGrid3D(A.getcommgrid3D()->GetWorld(), A.getcommgrid3D()->GetGridLayers(), A.getcommgrid3D()->GetGridRows(), A.getcommgrid3D()->GetGridCols(), A.isSpecial()));
     UDERO * localResultant = new UDERO(0, A.GetLayerMat()->seqptr()->getnrow(), divisions3d[A.getcommgrid3D()->GetRankInFiber()], 0);
     localResultant->ColConcatenate(toconcatenate);
     SpParMat3D<IU, NUO, UDERO> C3D(localResultant, grid3d, A.isColSplit(), A.isSpecial());
     return C3D;
 }
 

}