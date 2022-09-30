add_subdirectory(ASpT)

# Hack for now, forward up a level
set(ASpT_INCLUDE_DIRS "${ASpT_INCLUDE_DIRS}" PARENT_SCOPE)
set(ASpT_SRCS "${ASpT_SRCS}" PARENT_SCOPE)