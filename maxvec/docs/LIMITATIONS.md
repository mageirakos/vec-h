# Limitations

## Critical Missing Features

### FAISS integration
- we can't force the index to be in `cpu-pinned` memory because index is handled by FAISS
    - engine limitation we'll get much worse data movement performance probably (moving index around)

### CPU
- Hash Join with List column (acero problem)

### GPU
- Loading to GPU

## Missing Features

### FAISS
- we don't support insert/update/delete : we are not using add with ids in FAISS

### DAG Scheduler
- 'Plan Splitter' for batched execution to have independent plans after batched vector search for subsequent operations.

### GPU
- Arrow -> cuDF list conversion & cuDF -> Arrow list conversion
- loading Cagra from Disk into the GPU (on DGX it fails, open faiss issue, untested elsewhere)


## Untested
- Everything on GPU is untestedsince loading does not work.
- Large scale (largelistarray) <- over 6M vectors
- storing/loading indexes (besides cagra...)