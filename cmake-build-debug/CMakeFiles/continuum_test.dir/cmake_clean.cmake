file(REMOVE_RECURSE
  "continuum_test.pdb"
  "continuum_test"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/continuum_test.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
