
#include <petscmat.h>

static char help[] = "Read in a Symmetric matrix in MatrixMarket format (only the lower triangle). \n\
  Assemble it to a PETSc sparse SBAIJ (upper triangle) matrix. \n\
  Write it in a AIJ matrix (entire matrix) to a file. \n\
  Input parameters are:            \n\
    -fin <filename> : input file   \n\
    -fout <filename> : output file \n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
  PetscInt       i,m,n,nnz;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    *val,zero=0.0;
  FILE           *file;
  PetscViewer    view;
  int            *row,*col,*rownz;
  PetscBool      flg, permute;
  char           ordering[256] = MATORDERINGRCM;
  IS             rowperm       = NULL,colperm = NULL;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example does not work with complex numbers");
  ierr = PetscFinalize();
  return 0;
#endif

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Uniprocessor Example only\n");

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"Must indicate input file with -fin option");
  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  /* process header with comments */
  do {
    char *str = fgets(buf,PETSC_MAX_PATH_LEN-1,file);
    if (!str) SETERRQ(PETSC_COMM_SELF,1,"Incorrect format in file");
  }while (buf[0] == '%');

  /* The first non-comment line has the matrix dimensions */
  sscanf(buf,"%d %d %d\n",&m,&n,&nnz);
  ierr = PetscPrintf (PETSC_COMM_SELF,"m = %d, n = %d, nnz = %d\n",m,n,nnz);

  /* reseve memory for matrices */
  ierr = PetscMalloc4(nnz,&row,nnz,&col,nnz,&val,m,&rownz);CHKERRQ(ierr);
  for (i=0; i<m; i++) rownz[i] = 1; /* add 0.0 to diagonal entries */

  for (i=0; i<nnz; i++) {
    ierr = fscanf(file,"%d %d %le\n",&row[i],&col[i],(double*)&val[i]);
    if (ierr == EOF) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"i=%d, reach EOF\n",i);
    row[i]--; col[i]--;    /* adjust from 1-based to 0-based */
    rownz[col[i]]++;
  }
  fclose(file);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Read file completes.\n");CHKERRQ(ierr);

  /* Creat and asseble SBAIJ matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  //ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  //ierr = MatSeqSBAIJSetPreallocation(A,1,0,rownz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,0,rownz);CHKERRQ(ierr);

  /* Add zero to diagonals, in case the matrix missing diagonals */
  for (i=0; i<m; i++){
    ierr = MatSetValues(A,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0; i<nnz; i++) {
    ierr = MatSetValues(A,1,&col[i],1,&row[i],&val[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Assemble SBAIJ matrix completes.\n");CHKERRQ(ierr);

ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Poisson example options","");CHKERRQ(ierr);
  {
  permute          = PETSC_FALSE;
  ierr             = PetscOptionsFList("-permute","Permute matrix and vector to solving in new ordering","",MatOrderingList,ordering,ordering,sizeof(ordering),&permute);CHKERRQ(ierr);
  }
ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (permute) {
    Mat Aperm;
    double t0 = MPI_Wtime();
    ierr = MatGetOrdering(A,ordering,&rowperm,&colperm);CHKERRQ(ierr);
    double t1 = MPI_Wtime();
    ierr = MatPermute(A,rowperm,colperm,&Aperm);CHKERRQ(ierr);
    double t2 = MPI_Wtime();
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = Aperm;               /* Replace original operator with permuted version */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Permutation is performed. obtaining ordering time: %lf reordering time: %lf\n", t1-t0, t2-t1);CHKERRQ(ierr);
 }

  /* Write the entire matrix in AIJ format to a file */
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Write the entire matrix in AIJ format to file %s\n",fileout);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  }

  ierr = PetscFree4(row,col,val,rownz);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


