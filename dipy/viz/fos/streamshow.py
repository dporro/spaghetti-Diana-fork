import numpy as np

# fos modules
from fos import Actor
from fos.modelmat import screen_to_model
import fos.interact.collision as cll

# pyglet module
from pyglet.gl import *
from pyglet.lib import load_library

# dipy modules
from dipy.segment.quickbundles import QuickBundles
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle
from dipy.viz.colormap import orient2rgb
from dipy.tracking.metrics import downsample
from dipy.tracking.vox2track import track_counts

# other
import copy 
import cPickle as pickle

# trick for the bug of pyglet multiarrays
glib = load_library('GL')

# Tk dialogs
import Tkinter, tkFileDialog

# Pyside for windowing
from PySide.QtCore import Qt

# Interaction Logic:
from manipulator import Manipulator


question_message = """
>>>>Track Labeler
P : select/unselect the representative track.
E : expand/collapse the selected streamlines 
F : keep selected streamlines rerun QuickBundles and hide everything else.
A : select all representative streamlines which are currently visible.
I : invert selected streamlines to unselected
H : hide/show all representative streamlines.
>>>Mouse
Left Button: keep pressed with dragging - rotation
Scrolling :  zoom
Shift + Scrolling : fast zoom
Right Button : panning - translation
Shift + Right Button : fast panning - translation
>>>General
F1 : Fullscreen.
F2 : Next time frame.
F3 : Previous time frame.
F4 : Automatic rotation.
F12 : Reset camera.
ESC: Exit.
? : Print this help information.
"""


def streamline2rgb(streamline):
    """Compute orientation of a streamline and retrieve and appropriate RGB
    color to represent it.
    """
    # simplest implementation:
    return orient2rgb(streamline[0] - streamline[-1])


def compute_buffers(streamlines, alpha, save=False, filename=None):
    """Compute buffers for GL.
    """
    tmp = streamlines
    if type(tmp) is not type([]):
        tmp = streamlines.tolist()
    streamlines_buffer = np.ascontiguousarray(np.concatenate(tmp).astype('f4'))
    streamlines_colors = np.ascontiguousarray(compute_colors(streamlines, alpha))
    streamlines_count = np.ascontiguousarray(np.array([len(curve) for curve in streamlines],dtype='i4'))
    streamlines_first = np.ascontiguousarray(np.concatenate([[0],np.cumsum(streamlines_count)[:-1]]).astype('i4'))
    tmp = {'buffer': streamlines_buffer,
           'colors': streamlines_colors,
           'count': streamlines_count,
           'first': streamlines_first}
    if save:
        print "saving buffers to", filename
        np.savez_compressed(filename, **tmp)
        # This requires much more storage:
        # pickle.dump(tmp,
        #             open(filename, 'w'),
        #             protocol = pickle.HIGHEST_PROTOCOL)

        # Moreover using the gzip module to add compression makes it
        # extremely slow.
        # Moreover the npy/npz format takes care of endianess and
        # other archicetural tricky issues.
    return tmp # streamlines_buffer, streamlines_colors, streamlines_first, streamlines_count


def compute_buffers_representatives(buffers, representative_ids):
    """Compute OpenGL buffers for representatives from tractography
    buffers.
    """
    print "Creating buffers for representatives."
    count = buffers['count'][representative_ids].astype('i4')
    first = np.concatenate([[0], np.cumsum(count).astype('i4')])
    tmp = np.zeros(buffers['buffer'].shape[0], dtype=np.bool)
    for i, fid in enumerate(first):
        tmp[fid:fid+buffers['count'][i]] = True
        
    representative_buffers = {'buffer': np.ascontiguousarray(buffers['buffer'][tmp], dtype='f4'),
                              'colors': np.ascontiguousarray(buffers['colors'][tmp], dtype='f4'),
                              'count': np.ascontiguousarray(count),
                              'first': np.ascontiguousarray(first)}
    return representative_buffers


def compute_colors(streamlines, alpha):
    """Compute colors for a list of streamlines.
    """
    # assert(type(streamlines) == type([]))
    tot_vertices = np.sum([len(curve) for curve in streamlines])
    color = np.empty((tot_vertices,4), dtype='f4')
    counter = 0
    for curve in streamlines:
        color[counter:counter+len(curve),:3] = streamline2rgb(curve).astype('f4')
        counter += len(curve)

    color[:,3] = alpha
    return color


class StreamlineLabeler(Actor):   
    
    def __init__(self, name, buffers, clusters, representative_buffers=None, colors = None, vol_shape=None, representatives_line_width=5.0, streamlines_line_width=2.0, representatives_alpha=1.0, streamlines_alpha=1.0, affine=None, verbose=False):
        """StreamlineLabeler is meant to explore and select subsets of the
        streamlines. The exploration occurs through QuickBundles (qb) in
        order to simplify the scene.
        """
        super(StreamlineLabeler, self).__init__(name)

        if affine is None: self.affine = np.eye(4, dtype = np.float32)
        else: self.affine = affine      
         
        self.mouse_x=None
        self.mouse_y=None

        self.clusters = clusters
        self.representative_ids = self.clusters.keys()

        self.representatives_alpha = representatives_alpha

        # representative buffers:
        if representative_buffers is None:
            representative_buffers = compute_buffers_representatives(buffers, self.representative_ids)

        self.representatives_buffer = representative_buffers['buffer']
        self.representatives_colors = representative_buffers['colors']
        self.representatives_first = representative_buffers['first']
        self.representatives_count = representative_buffers['count']

        # full tractography buffers:
        self.streamlines_buffer = buffers['buffer']
        self.streamlines_colors = buffers['colors']
        self.streamlines_first = buffers['first']
        self.streamlines_count = buffers['count']

        print('MBytes %f' % (self.streamlines_buffer.nbytes/2.**20,))

        self.manipulator = Manipulator(self.clusters, clustering_function=None)

        self.expand = False
        self.hide_representatives = False

        self.representatives_line_width = representatives_line_width
        self.streamlines_line_width = streamlines_line_width

        self.vertices = self.streamlines_buffer # this is apparently requested by Actor

        # #buffer for selected virtual streamlines
        # self.selected = []
        # self.old_color = {}
        # self.hide_representatives = False
        # self.expand = False
        # self.verbose = verbose
        # self.streamlines_visualized_first = np.array([], dtype='i4')
        # self.streamlines_visualized_count = np.array([], dtype='i4')
        # self.history = [[self.clusters, self.streamlines, self.streamlines_ids, self.representatives_buffer, self.representatives_colors, self.representatives_first, self.representatives_count, self.streamlines_buffer, self.streamlines_colors, self.streamlines_first, self.streamlines_count]]
        # #shifting of streamline is necessary for dipy.tracking.vox2track.track_counts
        # #we also upsample using 30 points in order to increase the accuracy of streamline counts
        # self.vol_shape = vol_shape
        # if self.vol_shape !=None:
        #     self.representatives_shifted = [downsample(t+np.array(self.vol_shape)/2.,30) for t in self.representatives]

        # else:
        #     self.representatives_shifted = None




    def draw(self):
        """Draw virtual and real streamlines.

        This is done at every frame and therefore must be real fast.
        """
        glDisable(GL_LIGHTING)
        # representatives
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)        
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        # plot representatives if necessary:
        if not self.hide_representatives:
            glVertexPointer(3,GL_FLOAT,0,self.representatives_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.representatives_colors.ctypes.data)
            glLineWidth(self.representatives_line_width)
            glPushMatrix()
            if isinstance(self.representatives_first, tuple): print '>> first Tuple'
            if isinstance(self.representatives_count, tuple): print '>> count Tuple'
            glib.glMultiDrawArrays(GL_LINE_STRIP, 
                                   self.representatives_first.ctypes.data, 
                                   self.representatives_count.ctypes.data, 
                                   len(self.representatives_first))
            glPopMatrix()

        # plot tractography if necessary:
        if self.expand and self.streamlines_visualized_first.size > 0:
            glVertexPointer(3,GL_FLOAT,0,self.streamlines_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.streamlines_colors.ctypes.data)
            glLineWidth(self.streamlines_line_width)
            glPushMatrix()
            glib.glMultiDrawArrays(GL_LINE_STRIP, 
                                    self.streamlines_visualized_first.ctypes.data, 
                                    self.streamlines_visualized_count.ctypes.data, 
                                    len(self.streamlines_visualized_first))
            glPopMatrix()
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)      
        glLineWidth(1.)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)


    def process_mouse_position(self,x,y):
        self.mouse_x=x
        self.mouse_y=y


    def process_pickray(self,near,far):
        pass


    def update(self,dt):
        pass

    def select_streamline(self, ids):
        """Do visual selection of given representatives.
        """
        # WARNING: we assume that no streamlines can ever have color_selected as original color
        color_selected = np.array([1.0, 1.0, 1.0, 1.0], dtype='f4')
        if ids == 'all':
            ids = range(len(self.representatives))
        elif np.isscalar(ids):
            ids = [ids]
        for id in ids:
            if not id in self.old_color:
                self.old_color[id] = self.representatives_colors[self.representatives_first[id]:self.representatives_first[id]+self.representatives_count[id],:].copy()
                new_color = np.ones(self.old_color[id].shape, dtype='f4') * color_selected
                if self.verbose: print("Storing old color: %s" % self.old_color[id][0])
                self.representatives_colors[self.representatives_first[id]:self.representatives_first[id]+self.representatives_count[id],:] = new_color
                self.selected.append(id)

    def unselect_streamline(self, ids):
        """Do visual un-selection of given representatives.
        """
        if ids == 'all':
            ids = range(len(self.representatives))
        elif np.isscalar(ids):
            ids = [ids]
        for id in ids:
            if id in self.old_color:
                self.representatives_colors[self.representatives_first[id]:self.representatives_first[id]+self.representatives_count[id],:] = self.old_color[id]
                if self.verbose: print("Setting old color: %s" % self.old_color[id][0])
                self.old_color.pop(id)
                if id in self.selected:
                    self.selected.remove(id)
                else:
                    print('WARNING: unselecting id %s but not in %s' % (id, self.selected))
                    
    def invert_streamlines(self):
        """ invert selected streamlines to unselected
        """        
        tmp_selected=list(set(range(len(self.representatives))).difference(set(self.selected)))
        self.unselect_streamline('all')
        #print tmp_selected
        self.selected=[]
        self.select_streamline(tmp_selected)

    def process_messages(self,messages):
        msg=messages['key_pressed']
        #print 'Processing messages in actor', self.name, 
        #' key_press message ', msg
        if msg!=None:
            self.process_keys(msg,None)
        msg=messages['mouse_position']            
        #print 'Processing messages in actor', self.name, 
        #' mouse_pos message ', msg
        if msg!=None:
            self.process_mouse_position(*msg)

    def process_keys(self,symbol,modifiers):
        """Bind actions to key press.
        """
        prev_selected = copy.copy(self.selected)
        if symbol == Qt.Key_P:     
            print 'P'
            id = self.picking_representatives(symbol, modifiers)
            print('Streamline id %d' % id)
            if prev_selected.count(id) == 0:
                self.select_streamline(id)
            else:
                self.unselect_streamline(id)
            if self.verbose: 
                print 'Selected:'
                print self.selected

        if symbol==Qt.Key_E:
            print 'E'
            if self.verbose: print("Expand/collapse selected clusters.")
            if not self.expand and len(self.selected)>0:
                streamlines_selected = []
                for tid in self.selected: streamlines_selected += self.clusters.label2streamlinesids(tid)
                self.streamlines_visualized_first = np.ascontiguousarray(self.streamlines_first[streamlines_selected, :])
                self.streamlines_visualized_count = np.ascontiguousarray(self.streamlines_count[streamlines_selected, :])
                self.expand = True
            else:
                self.expand = False
        
        # Freeze and restart:
        elif symbol == Qt.Key_F and len(self.selected) > 0:
            print 'F'
            self.freeze()

        elif symbol == Qt.Key_A:
            print 'A'        
            print('Select/unselect all representatives')
            if len(self.selected) < len(self.representatives):
                self.select_streamline('all')
            else:
                self.unselect_streamline('all')
        
        elif symbol == Qt.Key_I:
            print 'I'
            print('Invert selection')
            print self.selected
            self.invert_streamlines()
            
        elif symbol == Qt.Key_H:
            print 'H'
            print('Hide/show representatives.')
            self.hide_representatives = not self.hide_representatives       
            
        elif symbol == Qt.Key_S:
            print 'S'
            print('Save selected streamlines_ids as pickle file.')
            self.streamlines_ids_to_be_saved = self.streamlines_ids
            if len(self.selected)>0:
                self.streamlines_ids_to_be_saved = self.streamlines_ids[np.concatenate([self.clusters.label2streamlinesids(tid) for tid in self.selected])]
            print("Saving %s streamlines." % len(self.streamlines_ids_to_be_saved))
            root = Tkinter.Tk()
            root.withdraw()
            pickle.dump(self.streamlines_ids_to_be_saved, 
                    tkFileDialog.asksaveasfile(), 
                    protocol=pickle.HIGHEST_PROTOCOL)

        elif symbol == Qt.Key_Question:
            print question_message
        elif symbol == Qt.Key_B:
            print 'B'
            print('Go back in the freezing history.')
            if len(self.history) > 1:
                self.history.pop()
                self.clusters, self.streamlines, self.streamlines_ids, self.representatives_buffer, self.representatives_colors, self.representatives_first, self.representatives_count, self.streamlines_buffer, self.streamlines_colors, self.streamlines_first, self.streamlines_count = self.history[-1]
                if self.reps=='representatives':
                    self.representatives=qb.representatives()
                if self.reps=='exemplars':
                    self.representatives, self.ex_ids = self.clusters.exemplars()#representatives()
                print len(self.representatives), 'representatives'
                self.selected = []
                self.old_color = {}
                self.expand = False
                self.hide_representatives = False

        elif symbol == Qt.Key_G:
            print 'G'
            print('Get streamlines from mask.')
            ids = self.maskout_streamlines()
            self.select_streamline(ids)

    def freeze(self):
        print("Freezing current expanded real streamlines, then doing QB on them, then restarting.")
        print("Selected representatives: %s" % self.selected)
        streamlines_frozen = []
        streamlines_frozen_ids = []
        for tid in self.selected:
            print tid
            part_streamlines = self.clusters.label2streamlines(self.streamlines, tid)
            part_streamlines_ids = self.clusters.label2streamlinesids(tid)
            print("virtual %s represents %s streamlines." % (tid, len(part_streamlines)))
            streamlines_frozen += part_streamlines
            streamlines_frozen_ids += part_streamlines_ids
        print "frozen streamlines size:", len(streamlines_frozen)
        print "Computing quick bundles...",
        self.unselect_streamline('all')
        self.streamlines = streamlines_frozen
        self.streamlines_ids = self.streamlines_ids[streamlines_frozen_ids] 
        
        root = Tkinter.Tk()
        root.wm_title('QuickBundles threshold')
        ts = ThresholdSelector(root, default_value=self.clusters.dist_thr/2.0)
        root.wait_window()
        
        self.clusters = QuickBundles(self.streamlines, dist_thr=ts.value, pts=self.clusters.pts)
        #self.clusters.dist_thr = qb.dist_thr/2.
        self.clusters.dist_thr = ts.value
        if self.reps=='representatives':
            self.representatives=qb.representatives()
        if self.reps=='exemplars':
            self.representatives,self.ex_ids = self.clusters.exemplars()
        print len(self.representatives), 'representatives'
        self.representatives_buffer, self.representatives_colors, self.representatives_first, self.representatives_count = self.compute_buffers(self.representatives, self.representatives_alpha)
        #compute buffers
        self.streamlines_buffer, self.streamlines_colors, self.streamlines_first, self.streamlines_count = self.compute_buffers(self.streamlines, self.streamlines_alpha)
        # self.unselect_streamline('all')
        self.selected = []
        self.old_color = {}
        self.expand = False
        self.history.append([self.clusters, 
                            self.streamlines, 
                            self.streamlines_ids, 
                            self.representatives_buffer, 
                            self.representatives_colors, 
                            self.representatives_first, 
                            self.representatives_count, 
                            self.streamlines_buffer, 
                            self.streamlines_colors, 
                            self.streamlines_first, 
                            self.streamlines_count])
        if self.vol_shape is not None:
            print("Shifting!")
            self.representatives_shifted = [downsample(t + np.array(self.vol_shape) / 2., 30) for t in self.representatives]
        else:
            self.representatives_shifted = None

    def picking_representatives(self, symbol,modifiers, min_dist=1e-3):
        """Compute the id of the closest streamline to the mouse pointer.
        """
        x, y = self.mouse_x, self.mouse_y
        # Define two points in model space from mouse+screen(=0) position and mouse+horizon(=1) position
        near = screen_to_model(x, y, 0)
        far = screen_to_model(x, y, 1)

        #print 'peak representatives ', near, far, x, y
        # Compute distance of representatives from screen and from the line defined by the two points above
        tmp = np.array([cll.mindistance_segment2track_info(near, far, xyz) \
                        for xyz in self.representatives])
        line_distance, screen_distance = tmp[:,0], tmp[:,1]
        if False: # basic algoritm:
            # Among the representatives within a range to the line (i.e. < min_dist) return the closest to the screen:
            closest_to_line_idx = np.argsort(line_distance)
            closest_to_line_thresholded_bool = line_distance[closest_to_line_idx] < min_dist
            if (closest_to_line_thresholded_bool).any():
                return closest_to_line_idx[np.argmin(screen_distance[closest_to_line_thresholded_bool])]
            else:
                return closest_to_line_idx[0]
        else: # simpler and apparently more effective algorithm:
            return np.argmin(line_distance + screen_distance)

    def maskout_streamlines(self):
        """ retrieve ids of representatives which go through the mask
        """
        mask = self.slicer.mask        
        #streamlines = self.streamlines_shifted
        streamlines = self.representatives_shifted
        #tcs,self.tes = track_counts(streamlines,mask.shape,(1,1,1),True)
        tcs,tes = track_counts(streamlines,mask.shape,(1,1,1),True)
        # print 'tcs:',tcs
        # print 'tes:',len(self.tes.keys())
        #find volume indices of mask's voxels
        roiinds=np.where(mask==1)
        #make it a nice 2d numpy array (Nx3)
        roiinds=np.array(roiinds).T
        #get streamlines going through the roi
        # print "roiinds:", len(roiinds)
        # mask_streamlines,mask_streamlines_inds=bring_roi_streamlines(streamlines,roiinds,self.tes)
        mask_streamlines_inds = []
        for voxel in roiinds:
            try:
                #mask_streamlines_inds+=self.tes[tuple(voxel)]
                mask_streamlines_inds+=tes[tuple(voxel)]
            except KeyError:
                pass
        mask_streamlines_inds = list(set(mask_streamlines_inds))
        print("Masked streamlines %d" % len(mask_streamlines_inds))
        print("mask_streamlines_inds: %s" % mask_streamlines_inds)
        return mask_streamlines_inds


class ThresholdSelector(object):
    def __init__(self, parent, default_value):
        self.parent = parent
        self.s = Tkinter.Scale(self.parent, from_=1, to=30, width=25, length=300, orient=Tkinter.HORIZONTAL)
        self.s.set(default_value)
        self.s.pack()
        self.b = Tkinter.Button(self.parent, text='OK', command=self.ok)
        self.b.pack(side=Tkinter.BOTTOM)
    def ok(self):
        self.value = self.s.get()
        self.parent.destroy()



