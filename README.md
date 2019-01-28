# AdamAccumulate

Adam optimizer with gradient accumulation for Keras deep learning library.

The optimizer has one additional parameter — ```accum_iters: integer >= 1``` — number of batches after which accumulated gradient is computed and weights are updated.

Example: if ```batch=32``` is too big (memory is not enough), try using ```batch=4``` and ```accum_iters=8```.
