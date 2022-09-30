# Poor man's FindCake, clean-up later

function(link_cake TARGET)
    if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/CAKE_on_CPU/libcake.so")
        target_include_directories(${TARGET}  PRIVATE ${PROJECT_SOURCE_DIR}/third_party/CAKE_on_CPU/include)
        target_link_directories(${TARGET}  PRIVATE ${PROJECT_SOURCE_DIR}/third_party/CAKE_on_CPU/)
        target_link_libraries(${TARGET}  cake)
    else()
        message(FATAL_ERROR "Cant find Cake please run install.sh")
    endif()
endfunction()