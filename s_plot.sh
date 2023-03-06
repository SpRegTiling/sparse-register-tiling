singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure7_to_9/post_process_results.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure7_to_9/plot_figure7.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure7_to_9/plot_figure8.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure7_to_9/plot_figure9.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure10/plot.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure12/plot.py
singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure13/plot.py
