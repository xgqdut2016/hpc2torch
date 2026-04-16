#pragma once
#include <iostream>
#include <sstream>
#include <cstdlib>

template <typename... Args>
inline void RuntimeCheckImpl(bool cond,
                             const char *file,
                             int line,
                             Args &&...args)
{
    if (!cond)
    {
        std::ostringstream oss;
        (oss << ... << args);

        std::cerr << "RuntimeCheck failed at "
                  << file << ":" << line << "\n"
                  << "  Reason: " << oss.str() << std::endl;
        std::abort();
    }
}

#define RUNTIME_CHECK(cond, ...) RuntimeCheckImpl((cond), __FILE__, __LINE__, __VA_ARGS__)
