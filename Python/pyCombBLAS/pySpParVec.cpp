#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVec.h"

using namespace std;

pySpParVec::pySpParVec(int64_t size)
{
	MPI::Intracomm comm = v.getCommGrid()->GetDiagWorld();
	
	int64_t locsize = 0;
	
	if (comm != MPI::COMM_NULL)
	{
		int nprocs = comm.Get_size();
		int dgrank = comm.Get_rank();
		locsize = (int64_t)floor(static_cast<double>(size)/static_cast<double>(nprocs));
		
		if (dgrank == nprocs-1)
		{
			// this may be shorter than the others
			locsize = size - locsize*(nprocs-1);
		}
	}

	SpParVec<int64_t, int64_t> temp(locsize);
	v = temp;
}


//pySpParVec::pySpParVec(const pySpParMat& commSource): v(commSource.A.commGrid);
//{
//}

//pySpParVec::pySpParVec(SpParVec<int64_t, int64_t> & in_v): v(in_v)
//{
//}

pyDenseParVec* pySpParVec::dense() const
{
	pyDenseParVec* ret = new pyDenseParVec(v.getnnz(), 0);
	ret->v += v;
	return ret;
}


int64_t pySpParVec::getnnz() const
{
	return v.getnnz();
}

void pySpParVec::add(const pySpParVec& other)
{
	v.operator+=(other.v);

	//return *this;
}

void pySpParVec::SetElement(int64_t index, int64_t numx)	// element-wise assignment
{
	v.SetElement(index, numx);
}


//const pySpParVec& pySpParVec::subtract(const pySpParVec& other)
//{
//	return *this;
//}

pySpParVec* pySpParVec::copy()
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v;
	return ret;
}


void pySpParVec::invert() // "~";  almost equal to logical_not
{
	v.Apply(invert64);
}


void pySpParVec::abs()
{
	v.Apply(abs64);
}

bool pySpParVec::any() const
{
	return getnnz() != 0;
}

bool pySpParVec::all() const
{
	return getnnz() == v.getTotalLength();
}

int64_t pySpParVec::intersectSize(const pySpParVec& other)
{
	cout << "intersectSize missing CombBLAS piece" << endl;
	return 0;
}

	
void pySpParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

void pySpParVec::printall()
{
	v.DebugPrint();
}


pySpParVec* pySpParVec::zeros(int64_t howmany)
{
	pySpParVec* ret = new pySpParVec(howmany);
	return ret;
}

pySpParVec* pySpParVec::range(int64_t howmany, int64_t start)
{
	pySpParVec* ret = new pySpParVec(howmany);
	ret->v.iota(howmany, start);
	return ret;
}



