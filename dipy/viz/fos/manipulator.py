import numpy as np
import random
import copy
import code


def clustering(streamline_ids, clusters_number, seed=0):
    """Create fake clustering just playing randomly with streamline
    ids. For testing purpose.
    """
    assert(clusters_number <= len(streamline_ids))
    random.seed(seed)
    representatives = set(random.sample(streamline_ids, clusters_number))
    what_remains = set(streamline_ids).difference(representatives)
    random_splits_ids = np.random.randint(low=0, high=len(representatives), size=len(what_remains))
    clusters = {}
    what_remains = np.array(list(what_remains))
    for i, representative in enumerate(representatives):
        clusters[representative] = set(what_remains[random_splits_ids==i].tolist()).union([representative])

    return clusters


class Manipulator(object):
    """This class provides the functions to manipulate streamline IDs
    that can suit an interactive session. It provides history
    capabilities and other amenities.
    """
    
    def __init__(self, initial_clusters, clustering_function):
        """Initialize the object.
        """
        self.initial_clusters = copy.deepcopy(initial_clusters)
        self.history = []
        self.clusters_reset(initial_clusters)
        self.clustering_function = clustering_function
        

    def clusters_reset(self, clusters):
        """Standard operations to do whenever self.clusters is
        modified.
        """
        self.clusters = clusters
        self.streamline_ids = reduce(set.union, clusters.values())
        self.representative_ids = set(self.clusters.keys())
        self.selected = set()
        self.expanded = set()
        self.show_representatives = True
        self.history.append('clusters_reset('+str(clusters)+')')
    

    def select(self, representative_id):
        """Select one representative.
        """
        assert(representative_id in self.representative_ids)
        assert(representative_id not in self.selected)
        self.selected.add(representative_id)
        self.history.append('select('+str(representative_id)+')')
        self.select_action(representative_id)


    def select_action(self, representative_id):
        """This is the actual action to perform in the application.
        """
        raise NotImplementedError


    def unselect(self, representative_id):
        """Select one representative.
        """
        assert(representative_id in self.representative_ids)
        assert(representative_id in self.selected)
        self.selected.remove(representative_id)
        self.history.append('unselect('+str(representative_id)+')')
        self.unselect_action(representative_id)
        

    def unselect_action(self, representative_id):
        """This is the actual action to perform in the application.
        """
        raise NotImplementedError


    def select_toggle(self, representative_id):
        """Toggle for dispatching select or unselect.
        """
        if representative_id not in self.selected:
            self.select(representative_id)
        else:
            self.unselect(representative_id)
        

    def select_all(self):
        """Select all streamlines.
        """
        # it is safer to make a copy of the ids to avoid subtle
        # dependecies when manipulating self.representative_ids in
        # later steps.
        self.selected = set(copy.deepcopy(self.representative_ids))
        self.history.append('select_all()')
        self.select_all_action()


    def select_all_action(self):
        raise NotImplementedError
        

    def unselect_all(self):
        self.selected = set()
        self.history.append('unselect_all()')
        self.unselect_all_action()
        

    def unselect_all_action(self):
        raise NotImplementedError
        

    def select_all_toggle(self):
        if self.selected == self.representative_ids:
            self.unselect_all()
        else:
            self.select_all()


    def remove_selected(self):
        """Remove all clusters whose representative is selected.
        """
        clusters = {}
        for representative in set(self.representative_ids).difference(self.selected):
            clusters[representative] = self.clusters[representative]
        self.clusters_reset(clusters)
        self.history.append('remove_selected()')
        self.remove_selected_action()


    def remove_selected_action(self):
        raise NotImplementedError
        

    def remove_unselected(self):
        """Remove all clusters whose representative is not selected.
        """
        clusters = {}
        for representative in self.selected:
            clusters[representative] = self.clusters[representative]
        self.clusters_reset(clusters)
        self.history.append('remove_unselected()')
        self.remove_unselected_action()


    def remove_unselected_action(self):
        raise NotImplementedError
        

    def recluster(self, clusters_number):
        """
        """
        streamline_ids_new = reduce(set.union, self.clusters.values())
        # sanity check:
        assert(clusters_number <= len(streamline_ids_new))
        clusters_new = self.clustering_function(streamline_ids_new, clusters_number)
        # sanity check:
        assert(streamline_ids_new == reduce(set.union, clusters_new.values()))
        self.history.append('recluster('+str(clusters_number)+')')
        self.clusters_reset(clusters_new)


    def invert(self):
        """Invert the selection of representatives.
        """
        self.selected = self.representative_ids.difference(self.selected)
        self.history.append('invert()')
        self.invert_action()


    def invert_action(self):
        raise NotImplementedError


    def show_representatives(self):
        """Show representatives.
        """
        self.show_representatives = True
        self.history.append('show_representatives()')
        

    def hide_representatives(self):
        """Do not show representatives.
        """
        self.show_representatives = False
        self.history.append('hide_representatives()')


    def expand_collapse_selected(self):
        """Toggle expand/collapse status of selected representatives.
        """
        self.expand = not self.expand
        self.history.append('expand_collapse_selected()')
        self.expand_collapse_selected_action()


    def expand_collapse_selected_action(self):
        raise NotImplementedError
        

    def replay_history(self, until=None):
        """Create a Manipulator object by replaying the history of
        self starting from self.initial_clusters.
        """
        m = Manipulator(self.initial_clusters, self.clustering_function)
        # skip the first action in the history because already done
        # during __init__():
        c = ['m.'+h for h in self.history[1:until]]
        c = '; '.join(c)
        c = code.compile_command(c)
        exec(c)
        return m
        

    def __str__(self):
        string = "Clusters: " + str(self.clusters)
        string += "\n"
        string += "Selected: " + str(self.selected)
        string += "\n"
        string += "Show Representatives: " + str(self.show_representatives)
        string += "\n"
        string += "Initial Clusters: " + str(self.initial_clusters)
        string += "\n"
        string += "History: " + str(self.history)
        return string


if __name__ == '__main__':

    seed = 0
    random.seed(0)
    np.random.seed(0)

    k = 3

    streamline_ids = np.arange(10, dtype=np.int)

    initial_clusters = clustering(streamline_ids, k)
    
    m = Manipulator(initial_clusters, clustering)
