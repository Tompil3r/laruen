
#ifndef RANGE_H
#define RANGE_H


namespace laruen::utils
{
    template <typename T> class Range
    {
        public:
            T start;
            T end;
            T step;

            Range(T start=0, T end=-1, T step=1);
    };
};


#endif