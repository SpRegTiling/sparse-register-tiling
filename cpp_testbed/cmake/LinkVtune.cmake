# Poor man's FindVtune, clean-up later

macro(link_vtune_if_available TARGET)
    if(EXISTS "/opt/intel/oneapi/vtune/latest/sdk/include/ittnotify.h")

        target_compile_definitions(${TARGET} PUBLIC -DVTUNE_AVAILABLE)
        target_include_directories(${TARGET}  PRIVATE /opt/intel/oneapi/vtune/latest/sdk/include)
        target_link_directories(${TARGET}  PRIVATE /opt/intel/oneapi/vtune/latest/sdk/lib64)

        # Each of the compiler options and libs we are linking against are quite important, so they will be detailed one at a time.
        target_compile_options(${TARGET}  PRIVATE
                -g                          # The most important, this compiles your application with profiling symbols.
                -fno-omit-frame-pointer     # This adds some extra information that allows the vtune application to more easily view your profile.
                -D_LINUX                    # This is suggested by Vtune, i don't know what this does.
                )

        target_link_libraries(${TARGET}  ittnotify dl)
    endif()
endmacro()