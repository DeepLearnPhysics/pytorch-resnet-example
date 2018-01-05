import os,sys,re


def make_training_plot( logfile, outputpath ):

    loglines = open(logfile,'r').readlines()

    # store tuples (epoch,loss,acc)
    test_pts  = []
    train_pts = []
    lr_pts    = []
    lr_max = 0
    lr_min = 1.0e6

    epoch_scale = 0.2
    
    current_epoch = 0
    for l in loglines:
        l = l.strip()
        data = l.split()
        if "train aveloss" in l:
            pt = ( int(filter(str.isdigit,data[1])), float(re.findall("\d+\.\d+",data[3])[0]), float(re.findall("\d+\.\d+",data[4])[0]) )
            current_epoch = pt[0]
            train_pts.append(pt)
        if "Test:Result*" in l:
            pt = ( current_epoch, float(data[4]), float(data[2]) )
            test_pts.append(pt)
        if "lr=" in l:
            pt = ( int(filter(str.isdigit,data[1])), float( data[-1].split("=")[-1] ) )
            if pt[1]>lr_max:
                lr_max = pt[1]
            if pt[1]<lr_min:
                lr_min = pt[1]
            lr_pts.append( pt )


    sys.argv.append("-b")
    import ROOT as rt
    rt.gStyle.SetOptStat(0)

    graphs = {}
    graphs["trainacc"]  = rt.TGraph( len(train_pts) )
    graphs["trainloss"] = rt.TGraph( len(train_pts) )
    graphs["testacc"]   = rt.TGraph( len(test_pts) )
    graphs["testloss"]  = rt.TGraph( len(test_pts) )
    graphs["lr"]        = rt.TGraph( len(lr_pts) )

    accmax = 0
    accmin = 1.0e6
    lossmax = 0
    lossmin = 1.0e6
    for ipt,pt in enumerate(train_pts):
        graphs["trainacc"].SetPoint( ipt, pt[0]*epoch_scale, pt[2] )
        graphs["trainloss"].SetPoint( ipt, pt[0]*epoch_scale, pt[1] )
        if accmax<pt[2]:
            accmax = pt[2]
        if accmin>pt[2]:
            accmin = pt[2]
        if lossmax<pt[1]:
            lossmax = pt[1]
        if lossmin>pt[1]:
            lossmin = pt[1]

    for ipt,pt in enumerate(test_pts):
        graphs["testacc"].SetPoint( ipt, pt[0]*epoch_scale, pt[2] )
        graphs["testloss"].SetPoint( ipt, pt[0]*epoch_scale, pt[1] )
        if accmax<pt[2]:
            accmax = pt[2]
        if accmin>pt[2]:
            accmin = pt[2]
        if lossmax<pt[1]:
            lossmax = pt[1]
        if lossmin>pt[1]:
            lossmin = pt[1]


    c = rt.TCanvas("c","",1400,600)
    c.Divide(2,1)

    # hitogram to set scales
    hloss = rt.TH1D("hloss",";epoch;loss",100, 0,train_pts[-1][0]*epoch_scale*1.1)
    hloss.SetMinimum( 0.5*lossmin )
    hloss.SetMaximum( 5.0*lossmax )

    hacc = rt.TH1D("hacc",";epoch;accuracy (percent)",100, 0,train_pts[-1][0]*epoch_scale*1.1)
    hacc.SetMinimum( 0.0 )
    hacc.SetMaximum( 100.0 )
    
    # Loss
    c.cd(1).SetLogy(1)
    c.cd(1).SetGridx(1)
    c.cd(1).SetGridy(1)
    hloss.Draw()
    graphs["trainloss"].SetLineColor(rt.kBlack)
    graphs["testloss"].SetLineColor(rt.kBlue)
    graphs["lr"].SetLineColor(rt.kRed)
    graphs["trainloss"].Draw("LP")
    graphs["testloss"].Draw("LP")

    # superimpose lr graph
    rightmax = 1.1*lr_max
    rightmin = 0.9*lr_min
    scale    = rt.gPad.GetUymax()/rightmax
    for ipt,pt in enumerate(lr_pts):
        graphs["lr"].SetPoint( ipt, pt[0]*epoch_scale, pt[1]*scale )    
    graphs["lr"].Draw("LPsame")
    lraxis = rt.TGaxis( rt.gPad.GetUxmax(), rt.gPad.GetUymin(), rt.gPad.GetUxmax(), rt.gPad.GetUymax(), rightmin, rightmax, 510, "+LG" )
    lraxis.SetLineColor(rt.kRed)
    lraxis.SetLabelColor(rt.kRed)
    lraxis.Draw()

    # Accuracy
    c.cd(2).SetLogy(0)
    c.cd(2).SetGridx(1)
    c.cd(2).SetGridy(1)
    hacc.Draw()
    graphs["trainacc"].SetLineColor(rt.kBlack)
    graphs["testacc"].SetLineColor(rt.kBlue)
    graphs["trainacc"].Draw("LP")
    graphs["testacc"].Draw("LP")
    
    c.Update()
    c.Draw()

    c.SaveAs(outputpath)


    
if __name__=="__main__":
    logfile = sys.argv[1]
    make_training_plot( logfile, "training.png" )
