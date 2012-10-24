import numpy as np
import nibabel as nib
from dipy.segment.quickbundles import QuickBundles
from dipy.viz.fos.streamshow import StreamlineLabeler
from dipy.viz.fos.streamwindow import Window
from dipy.viz.fos.guillotine import Guillotine
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import downsample
from fos import Scene
import pickle
from streamshow import compute_buffers, compute_buffers_representatives


def rotation_matrix(axis, theta_degree):
    theta = 1. * theta_degree * np.pi / 180.
    axis = 1. * axis / np.sqrt(np.dot(axis,axis))
    a = np.cos(theta / 2)
    b, c, d = - axis * np.sin(theta / 2)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


if __name__ == '__main__':
    
    subject = '05'
    num_M_seeds = 1
    directory_name='./'
    qb_threshold = 30 # in mm

    #load T1 volume registered in MNI space
    t1_filename = directory_name+'data/subj_'+subject+'/MPRAGE_32/T1_flirt_out.nii.gz'
    print "Loading", t1_filename
    img = nib.load(t1_filename)
    data = img.get_data()
    affine = img.get_affine()

    #load the tracks registered in MNI space
    tracks_basenane = directory_name+'data/subj_'+subject+'/101_32/DTI/tracks_gqi_'+str(num_M_seeds)+'M_linear'
    buffers_filename = tracks_basenane+'_buffers.npz'
    try:
        print "Loading", buffers_filename
        buffers = np.load(buffers_filename)
    except IOError:
        fdpyw = tracks_basenane+'.dpy'    
        dpr = Dpy(fdpyw, 'r')
        T = dpr.read_tracks()
        dpr.close() 
    
        # T = T[:2000]
        T = np.array(T, dtype=np.object)

        T = [downsample(t, 12) - np.array(data.shape[:3]) / 2. for t in T]
        axis = np.array([1, 0, 0])
        theta = - 90. 
        T = np.dot(T,rotation_matrix(axis, theta))
        axis = np.array([0, 1, 0])
        theta = 180. 
        T = np.dot(T, rotation_matrix(axis, theta))

        buffers = compute_buffers(T, alpha=1.0, save=True, filename=buffers_filename)
    
    # load initial QuickBundles with threshold qb_threshold
    fpkl = directory_name+'data/subj_'+subject+'/101_32/DTI/qb_gqi_3M_linear_'+str(qb_threshold)+'.pkl'
    try:
        print "Loading", fpkl
        qb = pickle.load(open(fpkl))
    except IOError:
        print "Computing QuickBundles."
        qb = QuickBundles(T, qb_threshold, qb_n_points)
        pickle.dump(open(fpkl, 'w'), qb)


    print "Create buffers for clusters."
    tmp, representative_ids = qb.exemplars()
    clusters = dict(zip(representative_ids, [qb.label2tracksids(i) for i, rid in enumerate(representative_ids)]))
    
    # create the interaction system for tracks 
    tl = StreamlineLabeler('Bundle Picker',
                           buffers, clusters)
    
    title = 'Streamline Interaction and Segmentation'
    w = Window(caption = title, 
                width = 1200, 
                height = 800, 
                bgcolor = (.5, .5, 0.9) )

    scene = Scene(scenename = 'Main Scene', activate_aabb = False)

    data = np.interp(data, [data.min(), data.max()], [0, 255])    
    guil = Guillotine('Volume Slicer', data)

    scene.add_actor(guil)
    scene.add_actor(tl)

    w.add_scene(scene)
    w.refocus_camera()

