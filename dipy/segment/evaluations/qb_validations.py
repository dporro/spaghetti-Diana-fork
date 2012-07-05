""" Validating QuickBundles
"""

import numpy as np
import dipy as dp
# track reading
from dipy.io.dpy import Dpy
# segmenation
from dipy.segment.quickbundles import QuickBundles
# visualization
from fos import Window, Region
from fos.actor import Axes, Text3D
from bundle_picker import TrackLabeler
# metrics 
from dipy.tracking.metrics import downsample
from dipy.tracking.distances import (bundles_distances_mam,
					bundles_distances_mdf,
					most_similar_track_mam)

from matplotlib.mlab import find

def load_data(id):
	ids=['02','03','04','05','06','08','09','10','11','12']
	filename =  'data/subj_'+ids[id]+'_lsc_QA_ref.dpy'
	dp=Dpy(filename,'r')
	tracks=dp.read_tracks()
	dp.close()
	return tracks

def show_qb_streamlines(tracks,qb):
	# Create gui and message passing (events)
	w = Window(caption = 'QB validation', 
		width = 1200, 
		height = 800, 
		bgcolor = (0.,0.,0.2) )
	# Create a region of the world of actors
	region = Region( regionname = 'Main',
			extent_min = np.array([-5.0, -5, -5]),
			extent_max = np.array([5, 5 ,5]))
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

def get_random_streamlines(tracks,N):	
	#qb = QuickBundles(tracks,dist,18)
	#N=qb.total_clusters()
	random_labels = np.random.permutation(np.arange(len(tracks)))[:N]
	random_streamlines = [tracks[i] for i in random_labels]
	return random_streamlines
		
def count_close_tracks(sla, slb, dist_thr=20):
        cnt_a_close = zeros(len(slb))
        for ta in sla:
            dta = bundles_distances_mdf([ta],slb)[0]
#            dta = bundles_distances_mam([ta],slb)[0]
            cnt_a_close += binarise(dta, dist_thr)
        return cnt_a_close

'''
coverage = # neighb tracks / #tracks 
         = cntT.sum()/len(T)

overlap = (cntT>1).sum()/len(T)

missed == (cntT==0).sum()/len(T)
'''

#virtuals/#tracks
        
'''
compare_streamline_sets(sla,slb,dist=20):
	d = bundles_distances_mdf(sla,slb)
	d[d<dist]=1
	d[d>=dist]=0
	return d 
'''

def binarise(D, thr):
#Replaces elements of D which are <thr with 1 and the rest with 0
        return 1*(np.array(D)<thr)

id=0

tracks=load_data(id)

track_subset_size = 50000

tracks=tracks[:track_subset_size]
print 'Streamlines loaded'
#qb=QuickBundles(tracks,20,18)
#print 'QuickBundles finished'
#print 'visualize/interact with streamlines'
#window,region,axes,labeler = show_qb_streamlines(tracks,qb)

qb = QuickBundles(tracks,20,18)
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
counts, Qfreq, Rfreq 
191 clusters

[[   0    2  668]
 [   1 1174  828]
 [   2 1905  720]
 [   3 1279  703]
 [   4  491  572]
 [   5  117  505]
 [   6   28  393]
 [   7    2  249]
 [   8    2  144]
 [   9    0  104]
 [  10    0   40]
 [  11    0   23]
 [  12    0   15]
 [  13    0   16]
 [  14    0    7]
 [  15    0    8]
 [  16    0    5]]

Now with thresholds (10,10) and 10k tracks
QB finished with 1830 clusters
[[   0    6 1696]
 [   1 4879 1957]
 [   2 3454 1694]
 [   3 1315 1448]
 [   4  292 1060]
 [   5   52  687]
 [   6    2  401]
 [   7    0  285]
 [   8    0  200]
 [   9    0  164]
 [  10    0  114]
 [  11    0   80]
 [  12    0   43]
 [  13    0   39]
 [  14    0   27]
 [  15    0   21]
 [  16    0   20]
 [  17    0   19]
 [  18    0   21]
 [  19    0   12]
 [  20    0    8]
 [  21    0    3]
 [  22    0    1]]
'''

