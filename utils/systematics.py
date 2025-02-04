import awkward
import numpy

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