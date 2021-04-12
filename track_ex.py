from xobjects import CLContext
from xtrack import Model, TrackJob
from xparticles import Particles

ctx = xobjects.CLContext(device="1.1")
particles = Particles(ctx, 100)

mod = Model.from_madx_files("lhc.seq", "optics.madx")

sequence = mod.sequence_to_xtrack("lhcb1", order=2, slices=2)

job = TrackJob(ctx, sequence, particles)

job.monitor("ip5", at="entry", slots=100, start=1, rolling=True)
job.monitor("mqxfa.1l1", at="exit", slots=10, start=1, stop=10)
job.monitor_all(at="exit", slots=100)
job.max_turns(10)
job.track()

print(job.out.motitor["ip5"])
print(job.out.motitor["all"])

for turn in range(1000000):
    job.max_turns = turn
    mod.globals["kq4.l5b1"] = 0.3 * sin(turns)
    job.track()
