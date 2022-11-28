#ifndef LINALG_H
#define LINALG_H

#include "array_types.hpp"

template <class T>
T dot(vec<T> a, vec<T> b)
{
    T sum(0);
    for (intptr_t i=0; i<a.length(); i++)
    {
        sum += a(i) * b(i);
    }
    return sum;
}

template <class T>
vec<T> matvec(matrix<T> a, vec<T> v)
{
    intptr_t nr = a.nrows();
    intptr_t nc = a.ncols();
    vec<T> w(nr);
    for (intptr_t i=0; i<nr; i++)
    {
        w(i) = 0;
    }
    for (intptr_t i=0; i<nr; i++)
        for (intptr_t j=0; j<nc; j++)
        {
            w(i) += a(i,j) * v(j);
        }
    return w;
}

template <class T>
matrix<T> matmul(matrix<T> a, matrix<T> b)
{
    intptr_t rowa = a.nrows();
    intptr_t cola = a.ncols();
    intptr_t colb = b.ncols();

    T* a_raw = a.raw_ptr();
    T* b_raw = b.raw_ptr();

    matrix<T> c(rowa, colb);
    T* c_raw = c.raw_ptr();
    for (intptr_t i=0; i<rowa; i++)
        for (intptr_t j=0; j<colb; j++)
        {
            c(i,j) = 0;
            for (intptr_t k=0; k<cola; k++)
            {
                c(i,j) += a(i,k)*b(k,j);
            }
        }
    return c;
}
#endif
