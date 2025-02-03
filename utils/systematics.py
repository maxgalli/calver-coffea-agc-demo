import awkward
import numpy
import dask.array as da
import dask_awkward as dak

# functions creating systematic variations
#def jet_pt_resolution(pt):
#    # normal distribution with 5% variations, shape matches jets
#    counts = ak.num(pt)
#    pt_flat = ak.flatten(pt)
#    # get size of flattened array
#    size = dak.num(pt_flat)
#    print(size)
#    #resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
#    #return ak.unflatten(resolution_variation, counts)
#    resolution_variation = da.random.normal(1, 0.05, size=pt_flat.size)
#    return ak.unflatten(resolution_variation, counts)

def rand_gauss(item):
    seeds = (
        awkward.typetracer.length_one_if_typetracer(item).to_numpy()[[0, -1]].view("i4")
    )
    randomstate = numpy.random.Generator(numpy.random.PCG64(seeds))

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, awkward.contents.NumpyArray) or not isinstance(
            layout, (awkward.contents.Content,)
        ):
            return awkward.contents.NumpyArray(
                randomstate.normal(loc=1, scale=0.05, size=len(layout)).astype(numpy.float32)
            )
        return None

    out = awkward.transform(
        getfunction,
        awkward.typetracer.length_zero_if_typetracer(item),
        behavior=item.behavior,
    )
    if awkward.backend(item) == "typetracer":
        out = awkward.Array(
            out.layout.to_typetracer(forget_length=True), behavior=out.behavior
        )

    assert out is not None
    return out