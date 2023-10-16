#!/usr/bin/python
from sys import argv, maxsize
import numpy as np
from collections import defaultdict
from math import ceil
from Dataset import Dataset
import operator



# Author of the following frequent itemset mining code: Christian Borgelt
# http://www.borgelt.net/eclat.html
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def report (iset, pexs, supp, data):
    '''Recursively report item sets with the same support.
iset    base item set to report (list of items)
pexs    perfect extensions of the base item set (list of items)
supp    (absolute) support of the item set to report
data    static recursion/output data as a list
        [ target, supp, min, max, maxx, count [, out] ]'''
    if not pexs:                # if no perfect extensions (left)
        data[5] += 1            # count the reported item set
        if len(data) < 7: return# check for a destination
        n = len(iset)           # check the item set size
        if (n < data[2]) or (n > data[3]): return
        if isinstance(data[6], [].__class__):
            data[6].append((tuple(iset), (supp,)))
        else:                   # report the current item set
            for i in iset: data[6].write(str(i)+' '), 
            data[6].write('('+str(supp)+')\n')
    else:                       # if perfect extensions to process
        for i in range(len(pexs)): # do include/exclude recursions
            report(iset+[pexs[i]], pexs[i+1:], supp, data)
            report(iset,           pexs[i+1:], supp, data)

#-----------------------------------------------------------------------

def closed (tracts, elim):
    '''Check for a closed item set.
tracts  list of transactions containing the item set
elim    list of lists of transactions for eliminated items
returns whether the item set is closed'''
    for t in reversed(elim):    # try to find a perfect extension
        if tracts <= t: return False
    return True                 # return whether the item set is closed

#-----------------------------------------------------------------------

def maximal (tracts, elim, supp):
    '''Check for a maximal item set.
tracts  list of transactions containing the item set
elim    list of lists of transactions for eliminated items
supp    minimum support of an item set
returns whether the item set is maximal'''
    for t in reversed(elim):    # try to find a frequent extension
        if sum([w for x,w in tracts & t]) >= supp: return False
    return True                 # return whether the item set is maximal

#-----------------------------------------------------------------------

def recurse (tadb, iset, pexs, elim, data):
    '''Recursive part of the eclat algorithm.
tadb    (conditional) transaction database, in vertical representation,
        as a list of item/transaction information, one per (last) item
        (triples of support, item and transaction set)
iset    item set (prefix of conditional transaction database)
pexs    set of perfect extensions (parent equivalent items)
elim    set of eliminated items (for closed/maximal check)
data    static recursion/output data as a list
        [ target, supp, min, max, maxx, count [, out] ]'''
    tadb.sort()                 # sort items by (conditional) support
    xelm = []; m = 0            # init. elim. items and max. support
    for k in range(len(tadb)):  # traverse the items/item sets
        s,i,t = tadb[k]         # unpack the item information
        if s > m: m = s         # find maximum extension support
        if data[0] in 'cm' and not closed(t, elim+xelm):
            continue            # check for a perfect extension
        #if data[0] in 'cm':     # check for a perfect extension
        #    x = set(iset +pexs +[j for r,j,u in tadb[k:]])
        #    y = set().intersection(*[u for u,w in t])
        #    if not (y <= x): continue
        proj = []; xpxs = []    # construct the projection of the
        for r,j,u in tadb[k+1:]:# trans. database to the current item:
            u = u & t           # intersect with subsequent lists
            r = sum([w for x,w in u])
            if   r >= s:       xpxs.append(j)
            elif r >= data[1]: proj.append([r,j,u])
        xpxs = pexs +xpxs       # combine perfect extensions and
        xset = iset +[i]        # add the current item to the set and
        n    = len(xpxs) if data[0] in 'cm' else 0
        r    = recurse(proj, xset, xpxs, elim+xelm, data) \
               if proj and (len(xset)+n < data[4]) else 0
        xelm += [t]             # collect the eliminated items
        if   data[0] == 'm':    # if to report only maximal item sets
            if r < data[1] and maximal(t, elim+xelm[:-1], data[1]):
                report(xset+xpxs, [], s, data)
        elif data[0] == 'c':    # if to report only closed  item sets
            if r < s: report(xset+xpxs, [], s, data)
        else:                   # if to report all frequent item sets
            report(xset, xpxs, s, data)
    return m                    # return the maximum extension support

#-----------------------------------------------------------------------

def eclat (tracts, target='s', supp=2, min=1, max=maxsize, out=0):
    '''Find frequent item set with the eclat algorithm.
tracts  transaction database to mine (mandatory)
        The database must be a list or a tuple of transactions;
        each transaction must be a list or a tuple of items.
        An item can be any hashable object.
target  type of frequent item sets to find     (default: 's')
        s/a   sets/all   all     frequent item sets
        c     closed     closed  frequent item sets
        m     maximal    maximal frequent item sets
supp    minimum support of an item set         (default: 2)
        (positive: percentage, negative: absolute number)
min     minimum number of items per item set   (default: 1)
max     maximum number of items per item set   (default: no limit)
out     output file or array as a destination  (default: None)
returns a list of pairs (i.e. tuples with two elements),
        each consisting of a tuple with a found frequent item set
        and a one-element tuple with its (absolute) support.'''
    supp = -supp if supp < 0 else int(ceil(0.01*supp*len(tracts)))
    if supp <= 0: supp = 1      # check and adapt the minimum support
    if max  <  0: max  = maxsize# and the maximum item set size
    if len(tracts) < supp:      # check whether any set can be frequent
        return out if isinstance(out, [].__class__) else 0
    tadb = dict()               # reduce by combining equal transactions
    for t in [frozenset(t) for t in tracts]:
        if t in tadb: tadb[t] += 1
        else:         tadb[t]  = 1
    tracts = tadb.items()       # get reduced trans. and collect items
    items  = set().union(*[t for t,w in tracts])
    tadb   = dict([(i,[]) for i in items])
    for t in tracts:            # collect transactions per item
        for i in t[0]: tadb[i].append(t)
    tadb = [[sum([w for t,w in tadb[i]]), i, set(tadb[i])]
            for i in tadb]      # build and filter transaction sets
    sall = sum([w for t,w in tracts])
    pexs = [i for s,i,t in tadb if s >= sall]
    tadb = [t for t in tadb if t[0] >= supp and t[0] < sall]
    maxx = max+1 if max < maxsize and target in 'cm' else max
    data = [target, supp, min, max, maxx, 0]
    if not isinstance(out, (0).__class__): data.append(out)
    r = recurse(tadb, [], pexs, [], data)
    if len(pexs) >= min:        # recursively find frequent item sets
        if   target == 'm':     # if to report only maximal item sets
            if r < supp: report(pexs, [], sall, data)
        elif target == 'c':     # if to report only closed  item sets
            if r < s:    report(pexs, [], sall, data)
        else:                   # if to report all frequent item sets
            report([], pexs, sall, data)  # report the empty item set
    if isinstance(out, [].__class__): return out
    return data[5]              # return (number of) found item sets

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------












class ItemsetMiner():
    def __init__(self, data, splits, minsupport, minitems, maxitems, selected_indices, show_ignored_attributes):
        self.data = data
        self.splits = splits
        self.minsupport = minsupport
        self.minitems = minitems
        self.maxitems = maxitems
        self.selected_indices = selected_indices
        self.itemsets = []
        self.show_ignored_attributes = show_ignored_attributes


    def beautify_names_and_normalize_counts(self, itemsets_and_counts):
        max_count = float(max(itemsets_and_counts, key=operator.itemgetter(1))[1])
        min_count = float(min(itemsets_and_counts, key=operator.itemgetter(1))[1])
        if min_count == max_count:
            min_count = 0.0
        max_count -= min_count
        ret = []
        for k in range(len(itemsets_and_counts)):
            i,c = itemsets_and_counts[k]
            name = str(i)[12:-3].replace("', '", ' & ').replace('_', ' ')
            if len(name) >= 80:
                return [('Patterns contain too many items', 1.0)]
            ret.append((name, int(70*(float(c)-min_count)/max_count)+30))
        return ret
        

    def build_itemsets(self, constraint='0', ignore_label=False):
        set_collection = []
        self.item_frequencies = defaultdict(int)
        for i,entry in enumerate(self.data.original_data[self.selected_indices]):
            out = []
            for j,amount in enumerate(entry[:-1]):
                if self.show_ignored_attributes == False:
                    if self.data.attribute_names[j] in self.data.ignored_attributes:
                        continue
                if amount > self.splits[j]:
                    if ignore_label == True:
                        if self.data.label_name == self.data.attribute_names[j]:
                            continue
                    self.item_frequencies[frozenset([self.data.attribute_names[j]])] += 1
                    out.append("%s" %(self.data.attribute_names[j].replace('-', '_').replace(' ', '_').replace("'", '')))
            if constraint == '0':
                set_collection.append(frozenset(out))
            elif constraint == '+':
                if self.data.original_data[self.selected_indices[i]][self.data.label_index] > 0:
                    set_collection.append(frozenset(out))
            elif constraint == '-':
                if self.data.original_data[self.selected_indices[i]][self.data.label_index] <= 0:
                    set_collection.append(frozenset(out))
        return set_collection


    def get_item_frequencies(self, transactions, top_k=10):
        # print self.item_frequencies
        return sorted(self.item_frequencies.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_k]

        # result = eclat(transactions, 's', -1, 1, 1, [])
        # d = defaultdict(list)
        # for r in result:
            # d[frozenset([r[0][0]])] = r[1][0]
        # return sorted(d.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_k]


    def get_frequent_itemsets(self, transactions, mode, top_k=10):
        result = eclat(transactions, mode, -self.minsupport, self.minitems, self.maxitems, [])
        d = defaultdict(list)
        for r in result:
            name = frozenset(r[0])
            d[name] = r[1][0]
        return sorted(d.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_k]


    def get_support(self, itemset, database):
        support = 0
        for record in database:
            if itemset.issubset(record):
                support += 1
        return support


    def get_support_set(self, itemset, database):
        support_set = []
        for i,record in enumerate(database):
            if itemset.issubset(record):
                support_set.append(i)
        return set(support_set)


    def get_delta_relevant_itemsets(self, delta, mode, top_k=10):
        data_pos = self.build_itemsets(constraint='+', ignore_label=True)
        data_neg = self.build_itemsets(constraint='-', ignore_label=True)
        closed_pos_raw = self.get_frequent_itemsets(data_pos, 'c', top_k=-1)
        if len(data_neg)*len(closed_pos_raw) == 0:
            return [(frozenset(["Select positive AND negative examples"]), 1.0)]

        closed_pos = {}
        for touple in closed_pos_raw:
            itemset = touple[0]
            closed_pos[itemset] = itemset

        # pre-calculate the positive and negative supports of all closed on the pos itemsets
        supports_neg = {}
        for cp in closed_pos:
            supports_neg[cp] = self.get_support(closed_pos[cp], data_neg)

        supports_pos = {}
        for cp in closed_pos:
            supports_pos[cp] = self.get_support(closed_pos[cp], data_pos)

        # pre-calculate the positive and negative support sets of all closed on the pos itemsets
        support_set_neg = {}
        for cp in closed_pos:
            support_set_neg[cp] = self.get_support_set(closed_pos[cp], data_neg)

        support_set_pos = {}
        for cp in closed_pos:
            support_set_pos[cp] = self.get_support_set(closed_pos[cp], data_pos)

        # remove the dominated itmesets, thus keeping all relevant itemsets
        keys = closed_pos.keys()
        relevant_sets = closed_pos.keys()
        for kx in keys:
            for ky in keys:
                try:
                    x = closed_pos[kx]
                    y = closed_pos[ky]
                    if x.issubset(y) and kx != ky:
                        if mode == 'absolute':
                            if len(support_set_neg[kx].difference(support_set_neg[ky])) <= delta:
                                relevant_sets.remove(ky)
                        elif mode == 'relative':
                            if len(support_set_neg[kx].difference(support_set_neg[ky])) <= delta_frac*(float(supports_neg[ky]) + float(supports_pos[ky])):
                                relevant_sets.remove(ky)
                except:
                    pass

        # find the top-k relevant sets
        qualities = {}
        p0 = float(len(data_pos))/float(len(data_neg) + len(data_pos))
        for r in relevant_sets:
            p = float(supports_pos[r])
            n = float(supports_neg[r])
            binomial_q = np.sqrt(p+n)*((p/(p+n)) - p0)
            qualities[r] = binomial_q

        return sorted(qualities.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_k]


    def get_subgroups(self, top_k=10):
        data_pos = self.build_itemsets(constraint='+', ignore_label=True)
        data_neg = self.build_itemsets(constraint='-', ignore_label=True)
        freq_itemsets_raw = self.get_frequent_itemsets(data_pos, 's', top_k=-1)

        freq_itemsets = {}
        for touple in freq_itemsets_raw:
            itemset = touple[0]
            freq_itemsets[itemset] = itemset
        # pre-calculate the positive and negative supports of all closed on the pos itemsets
        supports_neg = {}
        for f in freq_itemsets:
            supports_neg[f] = self.get_support(freq_itemsets[f], data_neg)

        supports_pos = {}
        for f in freq_itemsets:
            supports_pos[f] = self.get_support(freq_itemsets[f], data_pos)

        # find the top-k subgroups
        qualities = {}
        p0 = float(len(data_pos))/float(len(data_neg) + len(data_pos))
        for f in freq_itemsets:
            p = float(supports_pos[f])
            n = float(supports_neg[f])
            binomial_q = np.sqrt(p+n)*((p/(p+n)) - p0)
            qualities[f] = binomial_q

        return sorted(qualities.iteritems(), key=operator.itemgetter(1), reverse=True)[:top_k]


    def export_patterns(self, items, label_name):
        items = items[::-1]
        # preprocess the attribute names
        attributes = list(self.data.attribute_names)
        names = []
        all_names    = []
        for a in attributes:
            names.append(a.replace(' ', '_').replace('-', '_').replace("'", ''))
        # express the patterns as vectors
        all_vectors = []
        for item in items:
            binary_vector = np.zeros(len(names) - 1)
            name = ""
            for i in item[0]:
                name += "%s & " %(i)
                index = names.index(i)
                binary_vector[index] = 1.0
            all_vectors.append(binary_vector)
            all_names.append(name[:-3])
        all_vectors = np.array(all_vectors)
        nonzero_entries = np.nonzero(np.sum(all_vectors,axis=0))[0]
        # header
        out = "name"
        for i in nonzero_entries:
            out += ",%s" %(names[i])
        out += ",%s\n" %label_name.lower()
        # data
        for i,item in enumerate(items):
            out += "%s" %(all_names[i])
            for j in nonzero_entries:
                out += ",%.1f" %all_vectors[i][j]
            out += ",%.2f\n" %item[1]
        return out


    def export_patterns_extension_representation(self, items, label_name):
        # reverse the data, such that the ones with higher quality (originally listed first)
        # are now last and thus will be drawn on top of all others
        items = items[::-1]
        # get the supportsets
        dimensionality = len(self.data.data)
        attributes = []
        for a in self.data.attribute_names:
            attributes.append(a.replace(' ', '_').replace('-', '_').replace("'", ''))
        all_vectors  = []
        all_names    = []
        all_supports = []
        all_used_attribut_indices = set([])
        for pattern in items:
            pattern = pattern[0]
            supportset = set(range(dimensionality))
            name = ""
            for item_name in pattern:
                p = attributes.index(item_name)
                name += "%s & " %(item_name)
                column = set(np.nonzero(self.data.original_data.T[p])[0])
                supportset = column.intersection(supportset)
            all_used_attribut_indices = all_used_attribut_indices.union(supportset)
            all_supports.append(len(supportset))
            vector = np.zeros(dimensionality)
            vector[list(supportset)] = 1.0
            all_vectors.append(vector)
            all_names.append(name[:-3])
        # eliminate not supporting data records
        all_vectors = np.array(all_vectors)
        all_vectors = all_vectors.T[list(all_used_attribut_indices)].T
        # header
        out = "name"
        for i in list(all_used_attribut_indices):
            out += ",data_record_%d" %(i)
        out += ",%s\n" %label_name.lower()
        # data
        for i in range(len(all_vectors)):
            out += "%s" %(all_names[i])
            for v in all_vectors[i]:
                out += ",%.1f" %v
            out += ",%.1f\n" %all_supports[i]
        return out








if __name__ == "__main__":
    mode = 'b'
    discretization_type = '1'
    if len(argv) >= 3:
        discretization_type = argv[2]

    def generate_discretization_splits(data, discretization_type):
        splits = []
        if discretization_type == '1':
            splits = list(np.zeros(len(data.attribute_names)))
        elif  discretization_type == '2':
            for name in data.attribute_names:
                splits = list(np.average(data.data, axis=0))
        elif  discretization_type == '3':
            for name in data.attribute_names:
                stds = np.std(data.data, axis=0)
                splits = list(np.average(data.data, axis=0) + 2.5*stds)
        elif  discretization_type == '4':
            for name in data.attribute_names:
                splits = list(np.median(data.data, axis=0))
        return splits


    if mode == 'a':
        # DESCRIBE EACH DATA RECORD BY THE TOP-K PATTERNS    
        data = Dataset()
        data.read_in_data(argv[1])
        splits = list(np.zeros(len(data.attribute_names)))
        minsupport = 2
        minitems = 1
        maxitems = maxsize
        selected_indices = range(len(data.data))
        iMiner = ItemsetMiner(data, splits, minsupport, minitems, maxitems, selected_indices)

        # get the frequent itemsets
        transactions = iMiner.build_itemsets(constraint='0')
        items = iMiner.get_frequent_itemsets(transactions, 's', top_k=100)
        
        patternized_data = []
        for transaction in transactions:
            row = np.zeros(len(items))
            for i, pattern in enumerate(items):
                if pattern[0].issubset(transaction):
                    row[i] = 1.0
            patternized_data.append(row)

        out = "name"
        for i in items:
            pattern = str(i[0])[12:-3].replace("', '", ' and ')
            out += "," + pattern
        out += ",label"
        print(out)

        for i,p in enumerate(patternized_data):
            name = data.instance_names[i]
            pattern_vec = str(p)[2:-1].replace("  ", ',').replace(".", '.0').replace('\n', '')
            label = str(data.original_data[i][-1])
            print("%s,%s,%s" %(name,pattern_vec,label))


    elif mode == 'b':
        # BINARY VERSION OF THE TOP-K PATTERNS 
        data = Dataset()
        data.read_in_data(argv[1])
        splits = generate_discretization_splits(data, discretization_type)
        minsupport = 2
        minitems = 1
        maxitems = maxsize
        selected_indices = range(len(data.data))
        iMiner = ItemsetMiner(data, splits, minsupport, minitems, maxitems, selected_indices)

        # get the frequent itemsets
        transactions = iMiner.build_itemsets(constraint='0')
        items = iMiner.get_frequent_itemsets(transactions, 's', top_k=1000)
        iMiner.export_patterns(items, 'support')


