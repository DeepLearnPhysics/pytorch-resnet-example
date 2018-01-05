import os,sys,time

from plot_training import make_training_plot

logfile = "log_train_5a.txt"
outputpath = ""

while True:

    print "Updating %s from %s"%(outputpath, logfile)
    make_training_plot( logfile, outputpath )
    time.sleep(30)
