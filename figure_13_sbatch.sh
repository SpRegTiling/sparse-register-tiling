singularity exec -B ../:/datasets -B ../results:/results /usr/bin/rm -Rf /tmp/sp_reg_python/ # Rebuild every time to avoid ninja error
sh _figure_13_mr_4_6.sh
sh _figure_13_mr_8.sh