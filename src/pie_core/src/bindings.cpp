#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(pie_core, m)
{
    m.def("hello", []() { return "pie_core ✓"; });
}
