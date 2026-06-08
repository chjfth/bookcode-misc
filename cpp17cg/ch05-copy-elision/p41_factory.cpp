//********************************************************
// The following code example is taken from the book
//  C++17 - The Complete Guide
//  by Nicolai M. Josuttis (www.josuttis.com)
//  http://www.cppstd17.com
//
// The code is licensed under a
//  Creative Commons Attribution 4.0 International License
//  http://creativecommons.org/licenses/by/4.0/
//********************************************************

#include "factory.hpp"
#include <memory>
#include <atomic>

int main()
{
  int i = create<int>(42);

  std::unique_ptr<int> up = create<std::unique_ptr<int>>(new int{42});

  std::atomic<int> ai = create<std::atomic<int>>(42);
}
