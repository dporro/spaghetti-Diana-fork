""" Validating QuickBundles
"""
import os.path as osp
from time import time
import numpy as np
import dipy as dp
# track reading
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle, save_pickle
# segmenation
from dipy.segment.quickbundles import QuickBundles
# visualization
from fos import Window, Region
from fos.actor import Axes, Text3D, Line
from fos.actor.line import one_colour_per_line
from bundle_picker import TrackLabeler, track2rgb
from dipy.viz.colormap import boys2rgb
from dipy.viz import fvtk
# metrics 
from dipy.tracking.metrics import downsample, length
from dipy.tracking.distances import (bundles_distances_mam,
					bundles_distances_mdf,
					most_similar_track_mam)
from dipy.tracking.distances import approx_polygon_track
from nibabel import trackvis as tv
import colorsys
from matplotlib.mlab import find

def load_data(id):
	ids=['02','03','04','05','06','08','09','10','11','12']
	filename =  'data/subj_'+ids[id]+'_lsc_QA_ref.dpy'
	dp=Dpy(filename,'r')
        print 'Loading', filename
	tracks=dp.read_tracks()
	dp.close()
        #if down:
        #    return [downsample(t, 18) for t in tracks]
        return tracks

def load_data_elfthin(id, ref=True):
    ids=['02','03','04','05','06','08','09','10','11','12']
    dname='/media/SeaOfElfarion/PhD_thesis_data/Data/PROC_MR10032/'
    if ref:
        filename =  dname + 'subj_'+ids[id]+'/101_32/GQI/lsc_QA_ref.dpy'
    if not ref:
        filename =  dname + 'subj_'+ids[id]+'/101_32/GQI/lsc_QA.dpy'
    dp=Dpy(filename,'r')
    print 'Loading', filename
    tracks=dp.read_tracks()
    dp.close()
    #if down:
    #    return [downsample(t, 18) for t in tracks]
    return tracks

def load_dpy_save_qb(id, dist=10., down=18, ref=False):
    ids=['02','03','04','05','06','08','09','10','11','12']
    dname='/media/SeaOfElfarion/PhD_thesis_data/Data/PROC_MR10032/'
    if ref:
        filename =  dname + 'subj_'+ids[id]+'/101_32/GQI/lsc_QA_ref.dpy'
    if not ref:
        filename =  dname + 'subj_'+ids[id]+'/101_32/GQI/lsc_QA.dpy'
    dp=Dpy(filename,'r')
    print 'Loading', filename
    tracks=dp.read_tracks()
    dp.close()
    final='/home/eg309/Desktop/10subs/'
    if not ref:
        dist=dist/2.5
    qb=QuickBundles(tracks, dist, down)
    save_pickle(final+str(id)+'_dist_'+str(dist)+'_down_'+str(down)+'_ref_'+str(ref)+'.pkl', qb)

def load_qb(id, dist=10., down=18, ref=False):
    final='/home/eg309/Desktop/10subs/'
    return load_pickle(final+str(id)+'_dist_'+str(dist)+'_down_'+str(down)+'_ref_'+str(ref)+'.pkl')

def load_a_big_tractography_downsampled():
    if osp.exists('/tmp/3M_linear_12.npy'):
        return np.load('/tmp/3M_linear_12.npy')
    filename='/home/eg309/Data/trento_processed/subj_01/101_32/DTI/tracks_gqi_3M_linear.dpy'
    dp=Dpy(filename,'r')
    tracks=dp.read_tracks()
    dp.close()
    tracks=[downsample(t, 12) for t in tracks]
    np.save('/tmp/3M_linear_12.npy', tracks)
    return tracks

def load_pbc_data(id=None):
    if id is None:
        path = '/home/eg309/Data/PBC/pbc2009icdm/brain1/'
        streams, hdr = tv.read(path+'brain1_scan1_fiber_track_mni.trk')
        streamlines = [s[0] for s in streams]
        return streamlines
    if not osp.exists('/tmp/'+str(id)+'.pkl'):
        path = '/home/eg309/Data/PBC/pbc2009icdm/brain1/'
        streams, hdr = tv.read(path+'brain1_scan1_fiber_track_mni.trk')
        streamlines = [s[0] for s in streams]
        labels = np.loadtxt(path+'brain1_scan1_fiber_labels.txt')
        labels = labels[:,1]
        mask_cst = labels == id
        cst_streamlines = [s for (i,s) in enumerate(streamlines) if mask_cst[i]]
        save_pickle('/tmp/'+str(id)+'.pkl', cst_streamlines)
        return cst_streamlines
        #return [approx_polygon_track(s, 0.7853) for s in cst_streamlines]
    else:
        return load_pickle('/tmp/'+str(id)+'.pkl')    

def get_tractography_sizes():
        sizes = []
        for d in range(10):
                sizes.append(len(load_data(d)))
        return sizes

def show_qb_streamlines(tracks,qb):
	# Create gui and message passing (events)
	w = Window(caption='QB validation', 
		width=1200, 
		height=800, 
		bgcolor=(0.,0.,0.2) )
	# Create a region of the world of actors
	region = Region(regionname='Main', activate_aabb=False)
	# Create actors
	tl = TrackLabeler('Bundle Picker',
			qb,qb.downsampled_tracks(),
			vol_shape=(182,218,182),tracks_alpha=1)   
	ax = Axes(name = "3 axes", scale= 10, linewidth=2.0)
	vert = np.array( [[2.0,3.0,0.0]], dtype = np.float32 )
	ptr = np.array( [[.2,.2,.2]], dtype = np.float32 )
	tex = Text3D( "Text3D", vert, "(0,0,0)", 10*2.5, 10*.5, ptr)
	#Add actor to their region
	region.add_actor(ax)
	#region.add_actor(tex)
	region.add_actor(tl)
	#Add the region to the window
	w.add_region(region)
	w.refocus_camera()
	print 'Actors loaded'
	return w,region,ax,tex

def show_tracks_colormaps(tracks, qb, alpha=1):
    w = Window(caption='QuickBundles Representation', 
            width=1200, 
            height=800, 
            bgcolor=(0.,0.,0.2))
    region = Region(regionname='Main', activate_aabb=False)

    colormap = np.ones((len(tracks), 3))
    counter = 0
    for curve in tracks:
        colormap[counter:counter+len(curve),:3] = track2rgb(curve).astype('f4')
        counter += len(curve)
    colors = one_colour_per_line(tracks, colormap)
    colors[:,3]=alpha
    la = Line('Streamlines', tracks, colors, line_width=2)
    region.add_actor(la)
    w.add_region(region)
    w.refocus_camera()
    return w, region, la

def show_tracks_fvtk(tracks, qb=None, option='only_reps', r=None, opacity=1,
                     size=10, biggest_clusters=20):    
    if qb is None:
        colormap = np.ones((len(tracks), 3))
        for i, curve in enumerate(tracks):
            colormap[i] = track2rgb(curve)
        fvtk.add(r, fvtk.line(tracks,colormap, opacity=opacity, linewidth=3))
    else:
        centroids=qb.virtuals()
        if option == 'only_reps':
            colormap = np.ones((len(centroids), 3))
            H=np.linspace(0,1,len(centroids)+1)
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i],1.,1.))
                colormap[i] = col
            fvtk.add(r, fvtk.line(centroids, colormap, opacity=opacity,
                                  linewidth=5))
        if option == 'reps_and_tracks':
            colormap = np.ones((len(tracks), 3))
            H=np.linspace(0, 1, len(centroids)+1)
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i], 1., 1.))
                inds=qb.label2tracksids(i)
                colormap[inds]=col
            fvtk.add(r, fvtk.line(tracks, colormap, opacity=opacity, linewidth=3))
        if option == 'thick_reps':
            H=np.linspace(0,1,len(centroids)+1)
            S=np.array(qb.clusters_sizes())
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i],1.,1.))
                fvtk.add(r, fvtk.line([centroid], col, opacity=opacity,        
                    linewidth=np.interp(S[i],(S.min(),S.max()),(3,10))))
        if option == 'thick_reps_big':
            qb.remove_small_clusters(size)
            qb.virts=None
            centroids=qb.virtuals()
            H=np.linspace(0,1,len(centroids)+1)
            S=np.array(qb.clusters_sizes())

            for i, centroid in enumerate(centroids):
                #col=np.array(colorsys.hsv_to_rgb(H[i],1.,1.))
                col=track2rgb(centroid)
                #col=boys2rgb(centroid)
                fvtk.add(r, fvtk.line([centroid], col, opacity=opacity,
                                      linewidth=np.interp(S[i],(S.min(),S.max()),(3,10))))
        if option == 'big_cluster_tracks':
            qb.remove_small_clusters(size)
            qb.virts=None
            centroids=qb.virtuals()
            for i, centroid in enumerate(centroids):
                col=track2rgb(centroid)
                ctracks=qb.label2tracks(tracks, i)
                colormap = np.ones((len(ctracks), 3))
                colormap[:,:] = col
                fvtk.add(r, fvtk.line(ctracks, colormap, opacity=opacity, linewidth=3))
        if option == 'biggest_clusters':
            cs=qb.clusters_sizes()
            ci=np.argsort(cs)[:len(cs)-biggest_clusters]            
            qb.remove_clusters(ci)
            qb.virts=None
            centroids=qb.virtuals()
            for i, centroid in enumerate(centroids):
                col=track2rgb(centroid)
                ctracks=qb.label2tracks(tracks, i)
                colormap = np.ones((len(ctracks), 3))
                colormap[:,:] = col
                #fvtk.add(r, fvtk.line(ctracks, colormap, opacity=opacity, linewidth=3))
                fvtk.add(r, fvtk.line([centroid], col, opacity=opacity, linewidth=3))

    fvtk.show(r, size=(700, 700))

def get_random_streamlines(tracks,N):	
	#qb = QuickBundles(tracks,dist,18)
	#N=qb.total_clusters()
	random_labels = np.random.permutation(np.arange(len(tracks)))[:N]
	random_streamlines = [tracks[i] for i in random_labels]
	return random_streamlines
		
def count_close_tracks(sla, slb, dist_thr=20):
        cnt_a_close = np.zeros(len(slb))
        for ta in sla:
            dta = bundles_distances_mdf([ta],slb)[0]
            #dta = bundles_distances_mam([ta],slb)[0]
            cnt_a_close += binarise(dta, dist_thr)
        return cnt_a_close

tractography_sizes = [175544, 161218, 155763, 141877, 149272, 226456, 168833, 186543, 191087, 153432]

def split_halves(id):
        tracks = load_data(id)
        N = tractography_sizes[id]
        M = N/2
	first_half = np.random.permutation(np.arange(len(tracks)))[:M]
        second_half= np.random.permutation(np.arange(len(tracks)))[M:N]
        return [tracks[n] for n in first_half], [tracks[n] for n in second_half]

def dumped_ideas():
    """
    coverage = \# neighb tracks / \#tracks 
             = cntT.sum()/len(T)

    overlap = (cntT>1).sum()/len(T)

    missed == (cntT==0).sum()/len(T)
            
    compare_streamline_sets(sla,slb,dist=20):
            d = bundles_distances_mdf(sla,slb)
            d[d<dist]=1
            d[d>=dist]=0
            return d 
    """
    pass

def binarise(D, thr):
    #Replaces elements of D which are <thr with 1 and the rest with 0
    return 1*(np.array(D)<thr)

def half_split_comparisons():

    tractography_sizes = [175544, 161218, 155763, 141877, 149272, 226456, 168833, 186543, 191087, 153432]

    # size 02 175544

    id=0

    first, second = split_halves(id)

    print len(first), len(second)

    '''
    track_subset_size = 50000

    tracks=tracks[:track_subset_size]
    print 'Streamlines loaded'
    #qb=QuickBundles(tracks,20,18)
    #print 'QuickBundles finished'
    #print 'visualize/interact with streamlines'
    #window,region,axes,labeler = show_qb_streamlines(tracks,qb)
    '''

    downsampling = 12

    first_qb = QuickBundles(first,20,downsampling)
    n_clus = first_qb.total_clusters()
    print 'QB for first half has', n_clus, 'clusters'
    second_down = [downsample(s, downsampling) for s in second]

    '''
    random_streamlines={}
    for rep in [0]:
        random_streamlines[rep] = get_random_streamlines(qb.downsampled_tracks(), N)
    '''

    # Thresholded distance matrices (subset x tracks) where subset Q = QB centroids
    # and subset R = matched random subset. Matrices have 1 if the compared
    # tracks have MDF distance < threshold a,d 0 otherwise.
    #DQ=compare_streamline_sets(qb.virtuals(),qb.downsampled_tracks(), 20)
    #DR=compare_streamline_sets(random_streamlines[0],qb.downsampled_tracks(), 20)

    # The number of subset tracks 'close' to each track
    #neighbours_Q = np.sum(DQ, axis=0)
    #neighbours_R = np.sum(DR, axis=0)

    #neighbours_Q = count_close_tracks(qb.virtuals(), qb.downsampled_tracks(), 20)
    #neighbours_R = count_close_tracks(random_streamlines[0], qb.downsampled_tracks(), 20)

    neighbours_first = count_close_tracks(first_qb.virtuals(), first_qb.downsampled_tracks(), 20)
    neighbours_second = count_close_tracks(first_qb.virtuals(), second_down, 20)

    maxclose = np.int(np.max(np.hstack((neighbours_first,neighbours_second))))

    # The numbers of tracks 0, 1, 2, ... 'close' subset tracks
    counts = [(np.int(n), len(find(neighbours_first==n)), len(find(neighbours_second==n))) for n in range(maxclose+1)]

    print np.array(counts)

def prepare_timings():
    tracks=load_a_big_tractography_downsampled()
    blocks=np.arange(0,len(tracks),50000)[1:]
    print blocks
    distances=[30., 25., 20.]
    times={}
    noclusters={}
    for d in distances:
        times[d]=[]
        noclusters[d]=[]
        for b in blocks:
            t1=time()
            qb=QuickBundles(tracks[:b], d, None)
            t2=time()
            times[d].append(t2-t1)
            noclusters[d].append(qb.total_clusters())
            print d, b, t2-t1, qb.total_clusters()
    return times, noclusters, blocks

def plot_timings(times, noclusters, blocks, alpha=0.5, linewidth=4):
    import matplotlib.pyplot as plt
    
    #distances=[30., 25., 20.]
    distances=[20.,25.,30.]
    facecolors=['blue','red','green']
    ax1=plt.subplot(111)
    plt.rcParams['font.size']=20
    plt.rcParams['legend.fontsize']=20
    plt.title('Execution times for a range of \n tractography sizes and distance thresholds')
    for (i,d) in enumerate(distances):
        plt.plot(blocks, times[d], label=str(d)+' mm', color=facecolors[i], alpha=alpha, linewidth=linewidth)
        #plt.fill_between(blocks, times[d], facecolor=facecolors[i], alpha=alpha)
    ax1.set_xticklabels(['50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K', '450K', '500K'])
    ax1.set_xbound(50*10**3, 500*10**3)
    ax1.set_ylabel('Seconds')
    ax1.set_xlabel('Number of streamlines')
    plt.legend(loc='upper left')
    plt.show()

def plot_bas():
    import matplotlib.pyplot as plt
    bas10=load_pickle('/home/eg309/Desktop/bas_dist10.pkl')
    bas20=load_pickle('/home/eg309/Desktop/bas_dist20.pkl')
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    plt.rcParams['font.size']=24
    plt.rcParams['legend.fontsize']=20
    #plt.title('Bundle Adjacencies of the \n combinations of the 10 subjects')
    from itertools import combinations
    xticks=[]
    pairs=[]
    for c in combinations(range(10), 2):
        xticks.append(str(c[0]+1)+'-'+str(c[1]+1))
        pairs.append(c)
    X=np.arange(len(bas20))+1
    plt.plot(X, bas20, label='20 mm', linewidth=4)
    plt.plot(X, bas10, label='10 mm', linewidth=4)
    plt.fill_between(X, bas20, facecolor=(0.1, 0.1, 0.6))
    plt.fill_between(X, bas10, facecolor=(0.1, 0.6, 0.1))
    ax1.set_ybound(0.3, 1.)
    ax1.set_xbound(1, 45)
    ax1.set_xticks(np.arange(1, 46))
    ax1.set_ylabel('Bundle adjacency')
    ax1.set_xlabel('Combination Number')
    #ax1.set_xticklabels(xticks)
    xtickNames = plt.setp(ax1, xticklabels=xticks)
    plt.setp(xtickNames, rotation=60, fontsize=20)
    plt.legend(loc='upper left')
    #http://matplotlib.sourceforge.net/faq/howto_faq.html#automatically-make-room-for-tick-labels

    #fig.canvas.mpl_connect('draw_event', on_draw)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.1)
    #fig.subplots_adjust(right=0.1)

    plt.show()
    return xticks, X

def show_best_worse_bas():
    bas10=load_pickle('/home/eg309/Desktop/bas_dist10.pkl')
    bas20=load_pickle('/home/eg309/Desktop/bas_dist20.pkl')
    from itertools import combinations
    pairs=[]
    for c in combinations(range(10), 2):
        pairs.append(c)

    print 'bas10'
    print 'min',np.min(bas10), 'max', np.max(bas10), 'mean', np.mean(bas10), 'std', np.std(bas10) 
    print 'argmin', pairs[np.argmin(bas10)]
    print 'argmax', pairs[np.argmax(bas10)]

    print 'bas20'
    print 'min',np.min(bas20), 'max', np.max(bas20), 'mean', np.mean(bas20), 'std', np.std(bas20)
    print 'argmin', pairs[np.argmin(bas20)]
    print 'argmax', pairs[np.argmax(bas20)]

    return bas10, bas20, pairs





def bundle_adjacency(dtracks0, dtracks1, dist):

    d01=bundles_distances_mdf(dtracks0,dtracks1)    
    pair12=[]
    solo1=[]
    for i in range(len(dtracks0)):
        if np.min(d01[i,:]) < dist:
            j=np.argmin(d01[i,:])
            pair12.append((i,j))
        else:            
            solo1.append(dtracks0[i])
    pair12=np.array(pair12)
    
    pair21=[]
    solo2=[]
    for i in range(len(dtracks1)):
        if np.min(d01[:,i]) < dist:
            j=np.argmin(d01[:,i])
            pair21.append((i,j))
        else:
            solo2.append(dtracks1[i])
            
    pair21=np.array(pair21)
    
    return 0.5*(len(pair12)/np.float(len(dtracks0))+len(pair21)/np.float(len(dtracks1)))

def keep_biggest(qb, biggest_clusters):
    #qb=load_qb(id, dist=dist, down=down, ref=ref)
    cs=qb.clusters_sizes()
    ci=np.argsort(cs)[:len(cs)-biggest_clusters]            
    qb.remove_clusters(ci)
    qb.virts=None
    #centroids=qb.virtuals()

def compare_biggest(biggest=100, ba_dist=10.0):
    dist=10.0
    down=18
    ref=True
    from itertools import combinations
    bas=[]
    for c in combinations(range(10), 2):
        print c
        qb=load_qb(c[0], dist=dist, down=down, ref=ref)
        keep_biggest(qb, biggest)
        v0=qb.virtuals()
        qb=load_qb(c[1], dist=dist, down=down, ref=ref)
        keep_biggest(qb, biggest)
        v1=qb.virtuals()
        bas.append(bundle_adjacency(v0, v1, ba_dist))
    return bas

def show_fornix(distance=15):
    tracks=load_pbc_data(5)
    print 'Streamlines loaded'
    qb=QuickBundles(tracks, distance, 18)
    #print 'QuickBundles finished'
    #print 'visualize/interact with streamlines'
    #window, region, axes, labeler = show_qb_streamlines(tracks, qb)
    #w, region, la = show_tracks_colormaps(tracks,qb)
    print 'Total number of clusters', qb.total_clusters()
    print 'Cluster sizes', qb.clusters_sizes()
    options=['only_reps', 'reps_and_tracks', 'thick_reps']    
    ren = fvtk.ren()
    ren.SetBackground(1,1,1)
    show_tracks_fvtk(tracks, None, r=ren, opacity=0.2)
    fvtk.clear(ren)
    #show_tracks_fvtk(tracks, qb, option=options[0], r=ren, opacity=0.2)
    #fvtk.clear(ren)
    show_tracks_fvtk(tracks, qb, option=options[2], r=ren, opacity=0.8)
    fvtk.clear(ren)
    show_tracks_fvtk(tracks, qb, option=options[1], r=ren, opacity=0.2)

def show_arcuate(ren, label_id=1, opacity=0.4):
    tracks=load_pbc_data(label_id)
    qb=QuickBundles(tracks, 40, 18)
    centroids=qb.virtuals()
    DM=bundles_distances_mdf(centroids, qb.downsampled_tracks())
    DM=np.squeeze(DM)
    #ren=fvtk.ren()    
    v=np.interp(DM, [DM.min(), DM.max()], [0, 1])
    red=np.interp(v, [0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0],
                     [0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0])
    green=np.zeros(red.shape)
    blue=np.interp(v, [0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0],
                      [1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0.0])
    #blue=green
    colors=np.vstack((red, green, blue)).T
    ln=fvtk.line(tracks, colors, opacity=opacity, linewidth=3)
    fvtk.add(ren,ln)
    ren.SetBackground(1, 1, 1)
    fvtk.show(ren, size=(700, 700))
    fvtk.record(ren, n_frames=1, out_path='/tmp/pics/'+str(label_id)+'_',
                size=(700, 700),bgr_color=(1, 1, 1),magnification=2)

    fvtk.add(ren,fvtk.line(centroids, fvtk.yellow, opacity=1., linewidth=10))
    fvtk.rm(ren, ln)
    ln2=fvtk.line(tracks, colors, opacity=0, linewidth=3)
    fvtk.add(ren, ln2)
    fvtk.show(ren, size=(700, 700))
    fvtk.record(ren, n_frames=1, out_path='/tmp/pics/'+str(label_id)+'_centr_',
                size=(700,700), bgr_color=(1, 1, 1), magnification=2)

def show_brains(id,dist=10.,down=18, ref=False, remove=0.003, biggest=50):    
    #tracks=load_data_elfthin(id, False)
    #track_subset_size = subset
    #tracks=tracks[:track_subset_size]
    #tracks=load_pbc_data(3)
    #print 'Streamlines loaded'
    #qb=QuickBundles(tracks, dist, 18)
    qb=load_qb(id, dist=dist, down=down, ref=ref)
    tracks=qb.downsampled_tracks()
    #print 'QuickBundles finished'
    #print 'visualize/interact with streamlines'
    #window, region, axes, labeler = show_qb_streamlines(tracks, qb)
    #w, region, la = show_tracks_colormaps(tracks,qb)
    options=['only_reps', 'reps_and_tracks', 'thick_reps', 'thick_reps_big',
             'big_cluster_tracks', 'biggest_clusters']
    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1)
    show_tracks_fvtk(tracks, None, r=ren, opacity=.2)
    picsd='/home/eg309/Desktop/pics/'
    fvtk.record(ren,n_frames=1,out_path=picsd + str(id)+'a',size=(700, 700),bgr_color=(1,1,1))
    fvtk.clear(ren)
    #show_tracks_fvtk(tracks, qb, option=options[2], r=ren, opacity=0.8)
    #fvtk.clear(ren)
    #show_tracks_fvtk(tracks, qb, option=options[3], r=ren, opacity=0.8, size=remove*len(tracks))
    #fvtk.record(ren,n_frames=1,out_path=picsd + str(id)+'b',size=(700, 700),bgr_color=(1,1,1)) 
    #fvtk.clear(ren)
    show_tracks_fvtk(tracks, qb, option=options[5], r=ren, opacity=1, size=20,
                     biggest_clusters=biggest)
    fvtk.record(ren,n_frames=1,out_path=picsd + str(id)+'b',size=(700, 700),bgr_color=(1,1,1)) 
    fvtk.clear(ren)
    return qb

def stats():

    allc=[]
    alls=[]
    allcov=[]
    alllengths=[]

    for i in range(10):
        qb=load_qb(i, ref=True)
        lengths=[]
        print 'ID', i
        print '---Total clusters ', qb.total_clusters()
        allc.append(qb.total_clusters())
        print '---Total streamlines ', len(qb.tracksd)
        alls.append(len(qb.tracksd))
        cs=qb.clusters_sizes()
        ci=np.argsort(cs)[len(cs)-100:] #1500 for 85% coverage     
        coverage=[]
        virts=qb.virtuals()
        for c in ci:
            coverage.append(cs[c])
            lengths.append(length(virts[c]))

        print '---Coverage ', np.sum(coverage)
        print '---Coverage %', 100*np.sum(coverage)/np.float(len(qb.tracksd))
        allcov.append(100*np.sum(coverage)/np.float(len(qb.tracksd)))
        alllengths.append(lengths)
    
    print 'Mean Number of Clusters', np.mean(allc), np.std(allc)
    print 'Mean Number of Streamlines', np.mean(alls), np.std(alls)
    print 'Mean Number of Coverage', np.mean(allcov), np.std(allcov)
    AL=np.array(alllengths).ravel()
    print 'Mean Length in 100 Clusters', np.mean(AL), np.std(AL)




if __name__ == '__main__' :
    #qb=show_brains(3, 10000, 25, 0.005)
    pass
    """
    clusters_sizes=[]
    for i in range(10):
        qb=show_brains(i, 10000, 25, 0.003)
        clusters_sizes.append(qb.clusters_sizes)
    """


    """
    N=qb.total_clusters()
    print 'QB finished with', N, 'clusters'

    random_streamlines={}
    for rep in [0]:
        random_streamlines[rep] = get_random_streamlines(qb.downsampled_tracks(), N)
            
    # Thresholded distance matrices (subset x tracks) where subset Q = QB centroids
    # and subset R = matched random subset. Matrices have 1 if the compared
    # tracks have MDF distance < threshold a,d 0 otherwise.
    #DQ=compare_streamline_sets(qb.virtuals(),qb.downsampled_tracks(), 20)
    #DR=compare_streamline_sets(random_streamlines[0],qb.downsampled_tracks(), 20)

    # The number of subset tracks 'close' to each track
    #neighbours_Q = np.sum(DQ, axis=0)
    #neighbours_R = np.sum(DR, axis=0)
    neighbours_Q = count_close_tracks(qb.virtuals(), qb.downsampled_tracks(), 20)
    neighbours_R = count_close_tracks(random_streamlines[0], qb.downsampled_tracks(), 20)

    maxclose = np.int(np.max(np.hstack((neighbours_Q,neighbours_R))))

    # The numbers of tracks 0, 1, 2, ... 'close' subset tracks
    counts = [(np.int(n), len(find(neighbours_Q==n)), len(find(neighbours_R==n)))
              for n in range(maxclose+1)]

    print np.array(counts)

    # Typically counts_Q shows (a) very few tracks with 0 close QB
    # centroids, (b) many tracks with a small number (between 1 and 3?) close QB
    # tracks, and (c) few tracks with many (>3?) close QB tracks

    # By contrast counts_R shows (a) a large number of tracks with 0 close
    # R (random) neighbours, (b) fewer tracks with a small number of close R
    # tracks, and (c) a long tail showing how the R sample has over-sampled
    # in dense parts of the tractography, coming up with several rather
    # similar tracks. By contast the QB tracks are dissimilar by design - or
    # can be thought of as more evenly distributed in track space.

    # The output below was generated with subject 02, 5k tracks, and threshold 20.
    # Column 0 is the neighbour count, and Columns 1 and 2 are the
    # number of tracks with that neighbour count.

    # I suppose you could say this revealed some kind of sparseness for the
    # QB subset by comparison with the Random one
    """


