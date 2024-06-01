import os
import random
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

noofnetlistsaved = 0 #Counts the number of netlists having a loss of less than 2.00 at 27 deg. C, TOP_TT process and 0.9 V supply. 
losslist = []

#--------------------------To find the multiplier and account for it in the netlist -------------------------------
def strtoval(strval):               #To take care of the multiplier in the netlist
    val = float(strval[:-1])
    multiplier = strval[-1]
    if multiplier.isdigit():
        return getvalue(strval)
    else :
        if strval[-1] == "m":       #milli
            val = val * 1e-3
        elif strval[-1] == "u":     #micro
            val = val * 1e-6
        elif strval[-1] == "n":     #nano
            val = val * 1e-9
        elif strval[-1] == "p":     #pico
            val = val * 1e-12
        elif strval[-1] == "K":     #kilo
            val = val * 1e3
        elif strval[-1] == "M":
            val = val * 1e6
        elif strval[-1] == "G":
            val = val * 1e9
        else :
            val = val * 1
        return val

#-------------------------------------To open and read the NETLIST file----------------------------------------------
def getfile(file_name):
	with open(file_name) as f:   #Read the given netlist
		lines = f.readlines()
	return lines

#-----------------------------------To save the edited value onto the NETLIST file------------------------------------
def valtostr(ele,val):          #Save the line to write into the netlist later
    toreturn = ele+"="+str(val)
    return toreturn

#-----------------------------To get the editable editable parameters in NETLIST file ---------------------------------
parameter_dict = {}
loss_dict = {}
def getparameters(lines):
    global parameter_dict
    for line in lines:              #Store the parameters in a parameter dictionary
        if line[:10] == "parameters":
            for element in line.split()[1:]:
                [ele, val] = element.split("=")
                parameter_dict[ele] = strtoval(val)

#---------------------------------------- ReLU activation function ----------------------------------------------------
def ReLU(x):
      if x>0: 
            return x
      else :
            return 0
      
#-------------------------------------------------Loss Definiton ------------------------------------------------------
def loss(gain, S11, Idd, NF, iip3):
      global loss_dict

      #Defintion of all specifications 
      gain_exp = 10
      S11_exp = -10
      Idd_exp = 0
      NF_exp = 4
      iip3_exp = -15

      #Defining all the losses
      loss_gain = ReLU(gain_exp - gain)
      loss_S11 = ReLU(-S11_exp + S11)
      loss_Idd = ReLU(-Idd_exp + Idd)
      loss_NF = ReLU(-NF_exp + NF)
      loss_IIP3 = ReLU(iip3_exp - iip3)

      #Dictionary containing all losses used to calculate gradients
      loss_dict['loss_gain'] = loss_gain
      loss_dict['loss_S11'] = loss_S11
      loss_dict['loss_Idd'] = loss_Idd
      loss_dict['loss_NF'] = loss_NF
      loss_dict['loss_IIP3'] = loss_IIP3
      loss = loss_gain + loss_S11 + loss_Idd + loss_NF + loss_IIP3
      loss_dict['loss'] = loss
      return loss_dict

#-----------------------------------To return gradients - Perform Gradient Descent-----------------------------
def gradient(m, Rd, Ib): #Needs update here
      #Needs edit
      global loss_dict
      lr = 1
      dm = 0
      dRd = 0
      dIb = 0
      if loss_dict['loss_gain'] != 0:
            dRd = dRd - lr  
      if loss_dict['loss_Idd'] != 0 :
            dm = dm - lr
            dIb = dIb + lr*1e-6
      if loss_dict['loss_S11'] != 0:  
      	    dm = dm + lr * (20/(m-5))  
      grads = [dm, dRd, dIb]
      return grads

#----------------------------------Function to update current NETLIST parameters-------------------------------
def update(): #Needs update here
    #Whatever loss function and all needs to happen - happens here !!
    global parameter_dict, loss_dict
    #random_key = random.choice(['_gpar0', 'rd']) # Pick anyone at random and increase it by 2.5%
    #parameter_dict[random_key] = parameter_dict[random_key]*1.025
    m = parameter_dict['m1']
    Rf = parameter_dict['Rf']
    Ib = parameter_dict['Ib']
    grads = gradient(m, Rf, Ib)
    parameter_dict['m1'] = parameter_dict['m1'] - grads[0] 
    parameter_dict['m2'] = parameter_dict['m2'] - grads[0]   
    parameter_dict['Rf'] = parameter_dict['Rf'] - grads[1]
    parameter_dict['Ib'] = parameter_dict['Ib'] - grads[2]

#---------------------------------Function to update the NETLIST File------------------------------------------
def updatelines(lines):
    global parameter_dict
    for line in lines:              #Append to the list
        if line[:10] == "parameters":
            newline = line[:10]
            for i in range(1,len(line.split()[1:])+1):
                element = line.split()[i]
                [ele, val] = element.split("=")
                newele = valtostr(ele,parameter_dict[ele])
                element = newele
                newline = newline + " " + element
            newline = newline + "\n"
            lines[lines.index(line)] = newline
    return lines

#------------------------------------Function to write updated parameters into NETLIST file -------------------------
def writelines(lines, file_name):
    with open(file_name, "w") as f:
        f.writelines(lines)

#------------------------------------Function to run spectre from Python script --------------------------------------
def run_spectre(file_name):
    os.system("spectre "+file_name+" =log output.txt") #Run the code in terminal using spectre command

#------------------------------------Function to extract the DC operating point --------------------------------------
def DCop(file_name): #Update to add more details
    DCopdict = {}
    df = pd.read_csv(file_name, delim_whitespace=True, comment='*')
    DCopdict["vgs"] = df['V(Vout)'].values[0]
    DCopdict["ids"] = df['M0:ids'].values[0]
    DCopdict["gm"] = df['M0:gm'].values[0]
    DCopdict["gds"] = df['M0:gds'].values[0]
    DCopdict["vth"] = df['M0:vth'].values[0]
    DCopdict['vdsat'] = df['M0:vdsat'].values[0]
    DCopdict["Idd"] = abs(df['I(V0)'].values[0])
    return DCopdict

#------------------------------------Function to skip rows in the sp.out file ----------------------------------------
def skip_rows(total_rows):
   skip_list = [i for i in range(10)]  # Skip the first 8 rows
   skip_list.extend(range(11, total_rows, 2))  # Skip every even row after row 8
   return skip_list

#------------------------------------Function to extract S parameters - S11 and S21 ----------------------------------
def SP(file_name):
    total_rows = sum(1 for line in open(file_name))  # Count total rows
    df = pd.read_csv(file_name, skiprows = skip_rows(total_rows), header = None)
    df.columns = ['column1', 'column2', 'column3']
    freq = df['column1'].str.split(':').str[0].tolist()
    s11db = df['column1'].str.split(':').str[1].tolist()
    s11ph = df['column2'].str.split().str[0].tolist()
    s21db = df['column2'].str.split().str[1].tolist()
    s21ph = df['column3'].tolist()
    for i in range(len(freq)):
        freq[i] = float(freq[i])
        s11db[i] = float(s11db[i])
        s11ph[i] = float(s11ph[i])
        s21db[i] = float(s21db[i])
        s21ph[i] = float(s21ph[i])
    return freq, s11db, s11ph, s21db, s21ph

#-----------------------------------Function to extract Noise Figure (in dB) --------------------------------------------
def NFdb(file_name):
    nfdf = pd.read_csv(file_name, skiprows = 6, delim_whitespace=True)
    nfdf.columns = ['nfreq', 'NF']
    nfreq = nfdf['nfreq']
    nfdb = nfdf['NF']
    return nfreq, nfdb

#----------------------------Function that returns value if e is contained in its str representation ----------------
def getvalue(value_name):
    if 'e' in value_name:                     # Extracting the number before and after e
        num1=float(value_name.split('e')[0])
        num2=float(value_name.split('e')[1])
        num=num1*(10**num2)                   # Calculating the final number
    else:
        num=float(value_name)
    return num

#------------------------------Function to compare frequencies, used in IIP3 analysis -------------------------------
def check_freq(testfreq,targetfreq,fe):
	if testfreq<targetfreq+fe and testfreq>targetfreq-fe:
		return 1
	else:
		return 0
     
#------------------------------Function to get Vout, used in IIP3 analysis -----------------------------------------
def vout(lines):
	lines=lines.split()
	re=lines[1].split('(')[1]
	im=lines[2].split(')')[0]
	voutre=getvalue(re)
	voutim=getvalue(im)
	vout=np.sqrt(voutre*voutre+voutim*voutim)
	return vout

#-------------------------------Function to get magnitude of Vout, used in IIP3 analysis -----------------------------------
def getvoutmag(file_name):
	os.chdir("LNAhb.raw")
	lines=getfile(file_name)
	lines=lines[150:]
	os.chdir("..")
	f1=1.58e9
	f2=1.581e9
	f3=2*f2-f1
	fe=(f2-f1)/100
	flag=0
	flag_f1=0
	flag_f3=0
	flag_ft=0
	while True:
		if len(lines[0].split())<2:
			lines=lines[1:]
		elif 'freq' in lines[0].split()[0] and flag==0:
			flag=1
			lines=lines[1:]
		elif 'freq' in lines[0].split()[0] and flag==1:
			if flag_f1==0 and check_freq(float(lines[0].split()[1]),f2,fe)==1 :
				flag_ft=1
				while flag_ft==1:
					if 'Vout' in lines[0].split()[0]:
						flag_ft=0
						voutf1=vout(lines[0])
					else:
						lines=lines[1:]
				flag_f1=1
			elif flag_f3==0 and check_freq(float(lines[0].split()[1]),f3,fe)==1 :
				flag_ft=1
				while flag_ft==1:
					if 'Vout' in lines[0].split()[0]:
						flag_ft=0
						voutf3=vout(lines[0])
					else:
						lines=lines[1:]
				flag_f3=1
			lines=lines[1:]
			if flag_f1==1 and flag_f3==1:
				break
		else:
			lines=lines[1:]
	return voutf1,voutf3

#-------------------------------------Function to get the value of IIP3 --------------------------------------
def IIP3(voutf1m,voutf3m,pin):
	voutf1=20*np.log10(voutf1m)
	voutf3=20*np.log10(voutf3m)
	iip3=pin+(0.5*(voutf1-voutf3))
	return iip3

#---------------------------------------Function to get AC gain ----------------------------------------------------
def AC(file_name):
    df = pd.read_csv(file_name, delim_whitespace=True, skiprows = 7, comment= "*")
    df.columns =  ['freq', 're(V(Vout))', 'im(V(Vout))', 're(V(net2))', 'im(V(net2))']
    freq = df['freq'].tolist()
    reVout = df['re(V(Vout))'].tolist()
    imVout = df['im(V(Vout))'].tolist()
    reVin = df['re(V(net2))'].tolist()
    imVin = df['im(V(net2))'].tolist()
    Vout = np.array([complex(r, i) for r, i in zip(reVout, imVout)])
    Vin = np.array([complex(r, i) for r, i in zip(reVin, imVin)])
    gain = np.abs(Vout/Vin)
    phase = np.angle(Vout/Vin)
    gain_db = 20*np.log10(gain)
    return gain_db, phase, freq

#----------------------------------------Optimization Function --------------------------------------------------
def optimization():
    global parameter_dict, loss_dict
    netlist1 = "LNA.scs"
    netlist2 = "LNAhb.scs"
    dcout = "dc.out"
    acout = "ac.out"
    noiseout = "noise.out"
    spout = "sp.out"
    lines = getfile(netlist1)
    lineshb = getfile(netlist2)
    getparameters(lines)
    run_spectre(netlist1)
    run_spectre(netlist2)
    DCvals = DCop(dcout)
    Idd = DCvals['Idd']
    print(f"The DC operating point is \n gm : {DCvals['gm']} S, gds : {DCvals['gds']} S and VDSAT = {DCvals['vdsat']} V. \n The current drawn from VDD is Idc = {Idd} A \n Power Dissipated is {0.9*Idd} W")
    gain, _, freq = AC(acout)
    gainfreq = 0
    for i in range(len(freq)):
         if freq[i] == 1.58e9:
              gainfreq = gain[i]
              print(f"The AC gain at {freq[i]} (in dB) is {gain[i]}")
    #plt.figure()
    #plt.title("AC Gain")
    #plt.xlabel("Frequency")
    #plt.ylabel("Gain (in dB)")
    #plt.plot(freq, gain)
    #plt.show()
    nfreq, nfdb = NFdb(noiseout)
    for i in range(len(nfreq)) :
    	if nfreq[i] == 1.58e9:
    		NF = nfdb[i]
    		print(f"The NF (in dB) at {nfreq[i]} is {nfdb[i]}")
    #plt.figure()
    #plt.title("Noise Figure")
    #plt.xlabel("Frequency")
    #plt.ylabel("Noise Figure (in dB)")
    #plt.plot(nfreq, nfdb)
    #plt.show()
    freqs, s11db, _, s21db, _ = SP(spout) 
    s11freq = 0
    for i in range(len(freqs)) :
    	if freqs[i] == 1.58e9:
                s11freq = s11db[i]
                print(f"The S11 (in dB) at {freqs[i]} is {s11db[i]} and the S21 (in dB) is {s21db[i]}")
    #plt.figure()
    #plt.title("S11 (in dB)")
    #plt.xlabel("Frequency")
    #plt.ylabel("S11 (in dB)")
    #plt.plot(freqs, s11db)
    #plt.show()
    #plt.figure()
    #plt.title("S21 (in dB)")
    #plt.xlabel("Frequency")
    #plt.ylabel("S21 (in dB)")
    #plt.plot(freqs, s21db)
    #plt.show()
    hbout = "hbtest.fd.qpss_hb"
    vout1, vout3 = getvoutmag(hbout)
    iip3 = IIP3(vout1, vout3, parameter_dict['prf'])
    print(f"IIP3 for a Input power of {parameter_dict['prf']} is {iip3} dBm")
    loss_dict = loss(gainfreq, s11freq, Idd, NF, iip3)
    print(f"The value of loss is {loss_dict['loss']}")
    global noofnetlistsaved
    if loss_dict['loss'] < 0.11 :
    	netlistnew = "Netlist" + str(noofnetlistsaved) + ".scs"
    	netlistnewhb = "Netlisthb" + str(noofnetlistsaved) + ".scs"
    	noofnetlistsaved = noofnetlistsaved + 1
    	writelines(lines, netlistnew) 
    	writelines(lineshb, netlistnewhb)
    update()
    updatelines(lines)
    updatelines(lineshb)
    writelines(lines, netlist1)
    writelines(lineshb, netlist2)
    global losslist
    losslist.append(loss_dict['loss'])

#----------------------------------------Optimization Function without print--------------------------------------------------
def optimizationTemp():
    global parameter_dict, loss_dict
    dcout = "dc.out"
    acout = "ac.out"
    noiseout = "noise.out"
    spout = "sp.out"
    DCvals = DCop(dcout)
    Idd = DCvals['Idd']
    gain, _, freq = AC(acout)
    gainfreq = 0
    for i in range(len(freq)):
         if freq[i] == 1.58e9:
              gainfreq = gain[i]
    nfreq, nfdb = NFdb(noiseout)
    NF = 0
    for i in range(len(nfreq)) :
    	if nfreq[i] == 1.58e9:
    		NF = nfdb[i]
    freqs, s11db, _, s21db, _ = SP(spout) 
    s11freq = 0
    for i in range(len(freqs)) :
    	if freqs[i] == 1.58e9:
                s11freq = s11db[i]
    hbout = "hbtest.fd.qpss_hb"
    vout1, vout3 = getvoutmag(hbout)
    iip3 = IIP3(vout1, vout3, parameter_dict['prf'])
    loss_dict = loss(gainfreq, s11freq, Idd, NF, iip3)
    return loss_dict

#--------------------------------------Function to calculate sensitivity--------------------------------------------------
def sensitivity():
    file_name = "LNA.scs"
    lines = getfile(file_name)
    run_spectre(file_name)
    loss_i =  optimizationTemp()
    iloss = copy.copy(loss_i)
    global parameter_dict
    for parameter in parameter_dict:
         if parameter == 'm1' or parameter == 'm2' or parameter == 'Rf' or parameter == 'Ib':
              parameter_dict[parameter] = 1.01*parameter_dict[parameter]
    updatelines(lines)
    writelines(lines, file_name)
    run_spectre(file_name)
    loss_f = optimizationTemp()
    floss = copy.copy(loss_f)
    dloss = floss['loss'] - iloss['loss']
    for parameter in parameter_dict:
         if parameter == 'm1' or parameter == 'm2' or parameter == 'Rf' or parameter == 'Ib':
              parameter_dict[parameter] = parameter_dict[parameter]/1.01
    updatelines(lines)
    writelines(lines, file_name)
    return dloss

#--------------------------------------Function to get the process being run--------------------------------------------------
def getprocess(lines):
    global parameter_dict
    for line in lines:
        if line[0:7] == "include":
            elements = line.split()
            for element in elements:
                if element[0:7] == "section":
                    process = element.split("=")[1]
                    parameter_dict['process'] = process

#--------------------------------------Function to calculate Temperature sensitivity--------------------------------------------------
def tempsensitivity():
    file_name = "LNA.scs"
    lines = getfile(file_name)
    run_spectre(file_name)
    loss_i =  optimizationTemp()
    iloss = copy.copy(loss_i)
    global parameter_dict
    for parameter in parameter_dict:
         if parameter == 'circ_temp':
              parameter_dict[parameter] = 1.01*parameter_dict[parameter]
    updatelines(lines)
    writelines(lines, file_name)
    run_spectre(file_name)
    loss_f = optimizationTemp()
    floss = copy.copy(loss_f)
    dloss = floss['loss'] - iloss['loss']
    for parameter in parameter_dict:
         if parameter == 'circ_temp':
              parameter_dict[parameter] = parameter_dict[parameter]/1.01
    updatelines(lines)
    writelines(lines, file_name)
    return dloss

#--------------------------------------Function to update lines for PVT variations--------------------------------------------------
def updatelinesPVT(lines):
    global parameter_dict
    for line in lines:    #Append to the list
        if line[:7] == "include":
            newline = line[:-7]
            process = parameter_dict['process']
            newline = newline + process + "\n"
            lines[lines.index(line)] = newline
    return lines

#--------------------------------------Function to analyse loss at different temperatures-----------------------------------------------
loss_dict_temp = {}
def temp_analysis():
      netlist1 = "LNA.scs"
      netlist2 = "LNAhb.scs"
      global loss_dict_temp
      global parameter_dict
      lines = getfile(netlist1)
      lineshb = getfile(netlist2)
      run_spectre(netlist1)
      run_spectre(netlist2)
      temp_list = [-40, 0, 27, 100]
      for temp in temp_list:
            parameter_dict['circ_temp'] = temp
            updatelines(lines)
            writelines(lines, netlist1)
            updatelines(lineshb)
            writelines(lineshb, netlist2)
            run_spectre(netlist1)
            run_spectre(netlist2)
            loss_dict_temp[temp] = optimizationTemp()
            print(f"The loss at {temp} degree C is {loss_dict_temp[temp]['loss']}")
      parameter_dict['circ_temp'] = 27
      updatelines(lines)
      updatelines(lineshb)
      writelines(lines, netlist1)
      writelines(lineshb, netlist2)

#--------------------------------------Function to analyse loss at different processes-----------------------------------------------
loss_dict_process = {}
def process_analysis():
      netlist1 = "LNA.scs"
      netlist2 = "LNAhb.scs"
      global loss_dict_process
      global parameter_dict
      lines = getfile(netlist1)
      lineshb = getfile(netlist2)
      run_spectre(netlist1)
      run_spectre(netlist2)
      process_list = ['TOP_TT', 'TOP_FF', 'TOP_FS', 'TOP_SF', 'TOP_SS']
      for process in process_list:
            parameter_dict['process'] = process
            updatelinesPVT(lines)
            updatelinesPVT(lineshb)
            writelines(lines, netlist1)
            writelines(lineshb, netlist2)
            run_spectre(netlist1)
            run_spectre(netlist2)
            loss_dict_process[process] = optimizationTemp()
            print(f"The loss at {process} process is {loss_dict_process[process]['loss']}")
      parameter_dict['process'] = 'TOP_TT'
      updatelinesPVT(lines)
      updatelinesPVT(lineshb)
      writelines(lines, netlist1)
      writelines(lineshb, netlist2)

#--------------------------------------Function to analyse loss at different supply voltages-----------------------------------------------
loss_dict_voltage = {}
def voltage_analysis():
      netlist1 = "LNA.scs"
      netlist2 = "LNAhb.scs"
      global loss_dict_voltage
      global parameter_dict
      lines = getfile(netlist1)
      lineshb = getfile(netlist2)
      run_spectre(netlist1)
      run_spectre(netlist2)
      voltage_list = [0.81, 0.9, 0.99]
      for voltage in voltage_list:
            parameter_dict['VDD'] = voltage
            updatelines(lines)
            writelines(lines, netlist1)
            updatelines(lineshb)
            writelines(lineshb, netlist2)
            run_spectre(netlist1)
            run_spectre(netlist2)
            loss_dict_voltage[voltage] = optimizationTemp()
            print(f"The loss at {voltage}V is {loss_dict_voltage[voltage]['loss']}")
      parameter_dict['VDD'] = 0.9
      updatelines(lines)
      updatelines(lineshb)
      writelines(lines, netlist1)
      writelines(lineshb, netlist2)

#--------------------------------------Function to analyse loss at different temperatures-----------------------------------------------
def main():
    for _ in range(5):
        optimization()
        print(' ')
    a = list(range(1,6))
    global losslist
    b = losslist
    plt.plot(a,b)
    plt.xlabel("Iteration Number")
    plt.ylabel("Loss")
    plt.title("Resistive Feedback LNA (Loss Progression)")
    plt.show()
    for netlistidx in range(noofnetlistsaved):
    	netlist = "Netlist" + str(netlistidx) + ".scs"
    	netlisthb = "Netlisthb" + str(netlistidx) + ".scs"
    	print(f"For Netlist {netlistidx}, the analysis is done below -")
    	lines = getfile(netlist)
    	lineshb = getfile(netlisthb)
    	getparameters(lines)
    	writelines(lines, "LNA.scs")
    	writelines(lineshb, "LNAhb.scs")
    	run_spectre("LNA.scs")
    	run_spectre("LNAhb.scs")
    	temp_analysis()
    	print(' ')
    	process_analysis()
    	print(' ')
    	voltage_analysis()
    	print(' ')
    	dloss = sensitivity()
    	dtloss = tempsensitivity()
    	print(f"Sensitivity is {dloss}\nTemperature Sensitivity is {dtloss}")

main()
