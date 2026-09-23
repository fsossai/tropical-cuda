execute_process(
  COMMAND "${program}"
  RESULT_VARIABLE no_args_result
  OUTPUT_VARIABLE no_args_output
)
if(NOT no_args_result EQUAL 0 OR NOT no_args_output MATCHES "^Usage:")
  message(FATAL_ERROR "No arguments should print usage and exit successfully")
endif()

execute_process(
  COMMAND "${program}" graph.txt --unknown
  RESULT_VARIABLE unknown_result
  OUTPUT_VARIABLE unknown_output
  ERROR_VARIABLE unknown_error
)
if(unknown_result EQUAL 0 OR NOT unknown_output MATCHES "^Usage:" OR
   NOT unknown_error MATCHES "unknown option: --unknown")
  message(FATAL_ERROR "An unknown option should print usage and exit with an error")
endif()
