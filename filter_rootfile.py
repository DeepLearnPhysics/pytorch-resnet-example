from __future__ import print_function
import os,sys
import ROOT as rt
from larcv import larcv

infile = "/home/phy68/data/practice_test_5k.root"
outfile = "filtered.root"

io = larcv.IOManager(larcv.IOManager.kBOTH)
io.add_in_file(infile)
io.set_out_file(outfile)
io.initialize()

nentries = io.get_n_entries()
print("Number of entries: ",nentries)

nsaved = 0
for ientry in range(nentries):
    io.read_entry(ientry)

    print("------------------------------")
    print("entry ",ientry)

    selectme = False

    event_truth_data = io.get_data("particle","mctruth")
    particle_v = event_truth_data.as_vector()
    
    nparticles = particle_v.size()
    print("particle list:")
    for iparticle in range(nparticles):
        part = particle_v.at(iparticle)
        # part is instance of class Particle. for definition: ../larcv2/larcv/core/DataFormat/Particle.h
        # PDG codes: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
        print("  [%d] PDG:%d  E:%.1f GeV"%(iparticle,part.pdg_code(),part.energy_init()))
        if part.pdg_code()==11:
            # example, electron selection
            selectme = True
    
    if selectme:
        print("SAVED")
        nsaved += 1
        io.save_entry()

io.finalize()
print("Number saved: ",nsaved)
print("done")
