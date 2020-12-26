ret = [ ['reference', 'lambda', 'albedo', 'err', 'g', 'err'] ]

for sfile in scat_files:
    f   = open(sfile, 'r')
    ref = f.readline().rstrip()
    f.close()

    data = Table.read(sfile, format='ascii', header_start=1)
    for k in range(len(data)):
        scat_waves.append(data['wave,'][k])
        scat_albedo.append(data['albedo,'][k])
        scat_albedo_unc.append(data['delta,'][k])
        scat_g.append(data['g,'][k])
        scat_g_unc.append(data['delta'][k])
        scat_ref.append(ref)
        print(ref, data['wave,'][k], data['albedo,'][k], data['delta,'][k], data['g,'][k], data['delta'][k])
        ret.append( [ref, data['wave,'][k], data['albedo,'][k], data['delta,'][k], data['g,'][k], data['delta'][k]] )


print(ret)

with open(os.path.join(path, 'scat', 'data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(ret)
exit()