# caterpillar: Predicting how molecules cross the brood-brain barrier

## How to use:
`preprocessing/dataloader.py` defines the DataLoader class which can be used to concisely load in the data for the specific model to train. Initializing a DataLoader class takes up to 4 arguments:

- `filename`: specifies what the datasource is. Uses `pandas.read_csv()` to load in the file. The data loaded in must have the following columns: `smile` (SMILE Chemical Representation) and `p_np` (1 or 0). **Required**
- `fiftyfifty`: `True` or `False`. Specifies whether the data should be used as is or whether it should be subset such that p_np are of equal value. **Defaults to `False`.** *NOTE: Currently only works when (p_np == 1) > (p_np == 0)*.
- `modelType`: `"gnn"` or `"spectral"`. Specifies the data output. See the `getData()` method for more information. **Defaults to `"gnn"`**
- `addChemFeatures`: `True` or `False`. Only applicable if the modelType is `"spectral"`. If `True`, then various chemical features such as mass are added to the return of `getData()`. **Defaults to `False`.**
- `flatten`: `True` or False`. Only applicable if the modelType is `"gnn"`. If `True`, then `getData()` returns flattened adjacency matrices. **Defaults to `False`.**.
- `pad`: `True` or `False`. Only applicable if the modelType is `"gnn"`. If `True`, then `getData()` returns padded adjacency matrices. **Defaults to `True`.**.

`getData()`:
If `modelType` is `"gnn"`, returns `(x_train, y_train, x_test, y_test, longest_graph)` where the `x` are lists of adjacency matrices as numpy.ndarrays (flattened or unflattened depending on `flatten` initialization parameter) and `y` are lists of the corresponding `p_np`. `longest_graph` is the largest $n$ where the $n*n$ is the size of the adjacency matrix. Unless `pad` is `False`, the matrices are padded such that they are the same size.
If `modelType` is `"spectral"` returns `(x_train, y_train, x_test, y_test)` where `x` are lists of vectors containing the eigenvector and eigenvalues of the Laplacian. If `addChemFeatures` is `True` then other chemical features are added. The base size of a given element in `x` is 182 without chemical features, and 196 with.
