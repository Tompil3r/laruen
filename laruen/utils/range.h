
#ifndef RANGE_H
#define RANGE_H


namespace laruen::utils
{
    template <typename T> class Range
    {
        T start;
        T end;
        T step;


        public:
            Range(T start=0, T end=-1, T step=1);
    };
};


#endif