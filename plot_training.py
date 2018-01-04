import os,sys,re

logfile = sys.argv[1]

loglines = open(logfile,'r').readlines()

# store tuples (epoch,loss,acc)
test_pts  = []
train_pts = []


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


import ROOT as rt

graphs = {}
graphs["trainacc"] = rt.TGraph( len(train_pts) )
graphs["trainloss"] = rt.TGraph( len(train_pts) )
graphs["testacc"] = rt.TGraph( len(test_pts) )
graphs["testloss"] = rt.TGraph( len(test_pts) )
for ipt,pt in enumerate(train_pts):
    graphs["trainacc"].SetPoint( ipt, pt[0], pt[2] )
    graphs["trainloss"].SetPoint( ipt, pt[0], pt[1] )

for ipt,pt in enumerate(test_pts):
    graphs["testacc"].SetPoint( ipt, pt[0], pt[2] )
    graphs["testloss"].SetPoint( ipt, pt[0], pt[1] )



c = rt.TCanvas("c","",1400,600)
c.Divide(2,1)

# Loss
c.cd(1).SetLogy(1)
c.cd(1).SetGridx(1)
c.cd(1).SetGridy(1)
graphs["trainloss"].SetLineColor(rt.kBlack)
graphs["testloss"].SetLineColor(rt.kBlue)
graphs["trainloss"].Draw("ALP")
graphs["testloss"].Draw("LP")

# Accuracy
c.cd(2).SetLogy(0)
c.cd(2).SetGridx(1)
c.cd(2).SetGridy(1)
graphs["trainacc"].SetLineColor(rt.kBlack)
graphs["testacc"].SetLineColor(rt.kBlue)
graphs["trainacc"].Draw("ALP")
graphs["testacc"].Draw("LP")

c.Update()
c.Draw()

c.SaveAs("training.png")


    
