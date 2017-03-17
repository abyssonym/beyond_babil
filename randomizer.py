from randomtools.tablereader import (
    TableObject, get_global_label, tblpath, addresses)
from randomtools.utils import (
    classproperty, mutate_normal, shuffle_bits, get_snes_palette_transformer,
    write_multi, utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, run_interface, rewrite_snes_meta,
    clean_and_write, finish_interface)
from collections import defaultdict
from os import path
from time import time
from collections import Counter
import string


VERSION = 1
ALL_OBJECTS = None
LOCATION_NAMES = []

textdict = {}
reverse_textdict = {}
for line in open(path.join(tblpath, "font.txt")):
    character, byte = line.split("|")
    byte = chr(int(byte, 0x10))
    textdict[byte] = character
    reverse_textdict[character] = byte
assert ' ' in reverse_textdict


def bytestr_to_text(bytestr):
    text = "".join([textdict[c] if c in textdict else '~' for c in bytestr])
    return text


def text_to_bytestr(text):
    bytestr = "".join([reverse_textdict[c] for c in text])
    return bytestr


def get_location_names():
    if LOCATION_NAMES:
        return LOCATION_NAMES

    f = open(get_outfile(), "r+b")
    f.seek(addresses.location_names)
    for _ in xrange(128):
        s = ""
        while True:
            v = ord(f.read(1))
            if v == 0:
                break
            elif v  == 0xFF:
                s += " "
            elif 0x42 <= v <= 0x5b:
                s += string.uppercase[v-0x42]
            elif 0x5c <= v <= 0x75:
                s += string.lowercase[v-0x5c]
            elif 0x80 <= v <= 0x89:
                s += str(v - 0x80)
            elif v == 0xc0:
                s += "'"
            elif v == 0xc1:
                s += "."
            elif v == 0xc2:
                s += "-"
            elif v == 0xc4:
                s += "!"
            elif v == 0xc7:
                s += "&"
            else:
                #print "ALERT", hex(v)
                s += "~"
        LOCATION_NAMES.append(s)
    f.close()
    return get_location_names()


def write_location_names():
    global LOCATION_NAMES
    f = open(get_outfile(), "r+b")
    f.seek(addresses.location_names)
    for number in xrange(100):
        s = ""
        for c in "{0:0>2}".format(number):
            s += chr(int(c) + 0x80)
        s += chr(0)
        f.write(s)
    for c in "BOSS":
        f.write(chr(ord(c)+1))
    f.write(chr(0))
    f.close()
    LOCATION_NAMES = []
    get_location_names()


class CaveException(Exception): pass


class ReprSortMixin:
    def __hash__(self):
        return hash(self.__repr__())

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()


class ClusterExit(ReprSortMixin):
    def __init__(self, cluster, x, y):
        self.cluster = cluster
        self.x = x
        self.y = y
        if (self.mapid, x, y) in ClusterExit.exit_offsets:
            xx, yy = ClusterExit.exit_offsets[(self.mapid, x, y)]
            self.offset = (xx, yy)

    @property
    def xx(self):
        if hasattr(self, "offset"):
            return self.offset[0]
        return self.x

    @property
    def yy(self):
        if hasattr(self, "offset"):
            return self.offset[1]
        return self.y

    def __repr__(self):
        return "%x: %s %s" % (self.cluster.mapid, self.x, self.y)

    def is_adjacent(self, other):
        return (self.x-other.x) * (self.y-other.y) == 0 and (
            abs(self.x-other.x) == 1 or abs(self.y-other.y) == 1)

    @property
    def rank(self):
        return self.cluster.rank

    @property
    def mapid(self):
        return self.cluster.mapid

    @property
    def cluster_group(self):
        return self.cluster.cluster_group

    @classproperty
    def exit_offsets(self):
        if hasattr(ClusterExit, "_exit_offsets"):
            return ClusterExit._exit_offsets
        ClusterExit._exit_offsets = {}
        eopath = path.join(tblpath, "exit_offsets.txt")
        for line in open(eopath).readlines():
            line = line.strip()
            if not line:
                continue
            mapid, coords = line.split(":")
            mapid = int(mapid, 0x10)
            source, offset = coords.split()
            sx, sy = tuple(map(int, source.split(".")))
            ox, oy = tuple(map(int, offset.split(".")))
            ClusterExit._exit_offsets[mapid, sx, sy] = ox, oy
        return ClusterExit.exit_offsets

    def create_exit_trigger(self, mapid, x, y, use_given_mapid=False):
        assert mapid < 0xFE
        assert x < 32
        assert y < 32
        selfmap = MapObject.reverse_grid_index_canonical(self.mapid).index
        if use_given_mapid:
            othermap = mapid
        else:
            othermap = MapObject.reverse_grid_index_canonical(mapid).index
        for t in TriggerObject.getgroup(selfmap):
            if (t.x, t.y) == (self.x, self.y):
                # XXX: what to do about non-exit triggers?
                if t.is_exit:
                    # dangling exit
                    return
        t = TriggerObject.create_trigger(
            mapid=selfmap, x=self.x, y=self.y,
            misc1=othermap, misc2=(x|0x80), misc3=y)
        assert t.new_map == othermap
        assert t.dest_x == x
        assert t.dest_y == y


class Cluster(ReprSortMixin):
    def __init__(self, mapid, exits):
        self.mapid = mapid
        self.exits = []
        for (x, y) in sorted(exits):
            self.exits.append(ClusterExit(self, x, y))

    def __repr__(self):
        return "{0:0>3} -- ".format("%x" % self.mapid) + ", ".join(
            [str(x) for x in sorted(self.exits)])

    @staticmethod
    def from_string(s):
        mapid, exits = s.split(":")
        mapid = int(mapid, 0x10)
        exits = [tuple(map(int, x.split('.'))) for x in exits.split()]
        return Cluster(mapid, exits)

    @property
    def adjacencies(self):
        if hasattr(self, "_adjacencies"):
            return self._adjacencies
        adjacencies = [set([cx]) for cx in self.exits]
        while True:
            repeat = False
            for i, a in enumerate(adjacencies):
                for cx in self.exits:
                    if cx in a:
                        continue
                    for acx in sorted(a):
                        if cx.is_adjacent(acx):
                            adjacencies[i].add(cx)
                            repeat = True
            if not repeat:
                break
        adjacencies = set([tuple(sorted(a)) for a in adjacencies])

        self._adjacencies = sorted(adjacencies)
        return self.adjacencies

    @property
    def num_outs(self):
        return len(self.adjacencies)

    @property
    def health(self):
        return self.num_outs


class ClusterGroup:
    def __init__(self, clusters):
        self.clusters = sorted(set(clusters))

    @property
    def mapids(self):
        return set([c.mapid for c in self.clusters])

    @property
    def ranked_mapids(self):
        mapids = []
        for c in sorted(self.clusters, key=lambda c: c.rank):
            if c.mapid not in mapids:
                mapids.append(c.mapid)
        return mapids

    @property
    def health(self):
        health = 0
        for c in self.clusters:
            if c.num_outs <= 1:
                health -= 1
            elif c.num_outs >= 3:
                health += (c.num_outs - 2)
        return health

    @property
    def connected_exits(self):
        connected = set([])
        for (aa, bb) in self.connections:
            for x in list(aa) + list(bb):
                assert isinstance(x, ClusterExit)
                connected.add(x)
        return connected

    @property
    def exits(self):
        return set([x for c in self.clusters for x in c.exits])

    @property
    def unconnected_exits(self):
        return self.exits - self.connected_exits

    def connect(self):
        self.connections = []
        edge_candidates = [c for c in self.clusters if c.num_outs >= 2]
        edge_mapids = set([c.mapid for c in edge_candidates])
        if len(edge_mapids) < 2:
            raise CaveException("Not enough doors for two entrances to group")

        while True:
            start, finish = random.sample(edge_candidates, 2)
            if start.mapid != finish.mapid:
                break

        available_outs = dict([(c, c.num_outs) for c in self.clusters])
        available_outs[start] -= 1
        available_outs[finish] -= 1
        available_clusters = [start]
        remaining_clusters = [c for c in self.clusters
                              if c not in [start, finish]]
        random.shuffle(remaining_clusters)
        remaining_clusters.append(finish)

        while remaining_clusters:
            assert not set(available_clusters) & set(remaining_clusters)
            if not available_clusters:
                break
            max_index = len(available_clusters)-1
            a = random.randint(random.randint(0, max_index), max_index)
            a = available_clusters[a]
            candidates = None
            if len(available_clusters) == 1 and available_outs[a] == 1:
                candidates = [r for r in remaining_clusters
                              if available_outs[r] >= 2]
                if not candidates:
                    raise CaveException("Unable to connect all clusters.")
            else:
                candidates = remaining_clusters
            max_index = len(candidates)-1
            r = random.randint(0, random.randint(0, max_index))
            r = candidates[r]
            self.connections.append(tuple(sorted((a, r))))
            for c in (a, r):
                available_outs[c] -= 1
                assert available_outs[c] >= 0
            available_clusters.append(r)
            remaining_clusters.remove(r)
            available_clusters = [a for a in available_clusters
                                  if available_outs[a] > 0]

        if remaining_clusters:
            raise CaveException("Unable to connect all clusters.")
        self.start = start
        self.finish = finish
        self.available_outs = available_outs

    def rank_clusters(self):
        assert self.connections
        done = set([])
        for c in self.clusters:
            if c.mapid == self.start.mapid:
                c.rank = 0
                done.add(c)
        while len(done) < len(self.clusters):
            connections = list(self.connections)
            random.shuffle(connections)
            for a, b in self.connections:
                if ((a in done and b not in done) or
                        (b in done and a not in done)):
                    c = a if a not in done else b
                    rank = max([d.rank for d in done]) + 1
                    for e in self.clusters:
                        if e.mapid == c.mapid:
                            e.rank = rank
                            done.add(e)
                    break

    def fill_out(self):
        assert self.connections
        assert hasattr(self, "available_outs")
        assert [hasattr(c, "rank") for c in self.clusters]
        while True:
            remaining = sorted([r for r in self.clusters
                                if self.available_outs[r] > 0],
                               key=lambda c: (c.rank, c))
            if len(remaining) == 0:
                break
            elif len(remaining) == 1:
                r = remaining[0]
                if self.available_outs[r] == 1:
                    # dangling exit
                    self.connections.append((r, None))
                    self.available_outs[r] = 0
                    continue
                else:
                    pivot, partner = r, r
            else:
                by_available_outs = sorted(remaining,
                    key=lambda c: (self.available_outs[c], c))
                max_index = len(by_available_outs)-1
                pivot = by_available_outs[
                    random.randint(random.randint(0, max_index), max_index)]
                index = max(0, remaining.index(pivot)-1)
                remaining.remove(pivot)
                max_index = len(remaining)-1
                partner = random.randint(0, max_index)
                if partner <= index:
                    partner = random.randint(partner, index)
                else:
                    partner = random.randint(index, partner)
                partner = remaining[partner]
            self.connections.append(tuple(sorted((pivot, partner))))
            for c in [pivot, partner]:
                self.available_outs[c] -= 1
                assert self.available_outs[c] >= 0

    def specify(self):
        new_connections = []
        done = set([])
        for i, (a, b) in enumerate(self.connections):
            if b is None:
                ranked_clusters = sorted(self.clusters,
                                         key=lambda c: (c.rank, c))
                index = ranked_clusters.index(a)
                ranked_clusters = [r for r in ranked_clusters
                                   if r not in (self.start, self.finish)]
                max_index = len(ranked_clusters)-1
                index = max(0, min(index-1, max_index))
                b = random.randint(0, max_index)
                if index <= b:
                    b = random.randint(index, b)
                else:
                    b = random.randint(b, index)
                b = ranked_clusters[b].adjacencies
            else:
                b = [bb for bb in b.adjacencies if bb not in done]
            a = [aa for aa in a.adjacencies if aa not in done]
            a = random.choice(a)
            if b[0][0].mapid == a[0].mapid and len(b) > 1:
                # stop adjacencies from linking to themselves
                b = [bb for bb in b if bb != a]
            b = random.choice(b)
            new_connections.append((a, b))
            done.add(a)
            done.add(b)

        start = [a for a in self.start.adjacencies if a not in done]
        assert len(start) == 1
        self.start = start[0]
        finish = [a for a in self.finish.adjacencies if a not in done]
        assert len(finish) == 1
        self.finish = finish[0]
        done.add(self.start)
        done.add(self.finish)
        for c in self.clusters:
            for a in c.adjacencies:
                assert a in done

        self.connections = new_connections
        assert self.unconnected_exits == (set([x for x in self.start]) |
                                          set([x for x in self.finish]))

    def full_execute(self):
        for _ in xrange(10):
            try:
                self.connect()
                break
            except CaveException:
                pass
        else:
            raise CaveException("Excessive attempts to connect clusters.")
        self.rank_clusters()
        self.fill_out()
        self.specify()

    def create_exit_triggers(self):
        for aa, bb in self.connections:
            ax = sum([a.xx for a in aa]) / len(aa)
            ay = sum([a.yy for a in aa]) / len(aa)
            bx = sum([b.xx for b in bb]) / len(bb)
            by = sum([b.yy for b in bb]) / len(bb)
            amap = aa[0].mapid
            bmap = bb[0].mapid
            for a in aa:
                a.create_exit_trigger(bmap, bx, by)
            for b in bb:
                b.create_exit_trigger(amap, ax, ay)


def assign_formations(cluster_groups):
    print "ASSIGNING FORMATIONS"
    formations = [f for f in FormationObject.every if f.rank > 0]
    random.shuffle(formations)
    signatures = set([])
    temp = []
    for f in formations:
        if f.signature not in signatures:
            temp.append(f)
            signatures.add(f.signature)
    formations = temp
    monsters = sorted(set(m for f in formations for m in f.monsters))
    random.shuffle(monsters)
    done_monsters = set([])
    new_formations = []
    for m in monsters:
        if m in done_monsters:
            continue
        candidates = [f for f in formations if m in f.monsters]
        candidates = sorted(candidates, key=lambda f: f.rank)
        max_index = len(candidates)-1
        index = random.randint(random.randint(0, max_index), max_index)
        chosen = candidates[index]
        done_monsters |= set(chosen.monsters)
        new_formations.append(chosen)
    new_formations.append(FormationObject.get(0x1b7))  # Zeromus
    zeromon1, zeromon2 = MonsterObject.get(0xc8), MonsterObject.get(0xc9)
    assert zeromon1 in new_formations[-1].monsters
    assert zeromon2 in new_formations[-1].monsters
    unused_formations = [f for f in formations if f not in new_formations]
    unused_formations = sorted(unused_formations, key=lambda f: f.rank)
    while len(new_formations) < 256:
        max_index = len(unused_formations)-1
        index = random.randint(random.randint(0, max_index), max_index)
        chosen = unused_formations[index]
        new_formations.append(chosen)
        unused_formations.remove(chosen)
    unused_formations = [f for f in FormationObject.every
                         if f not in new_formations and f.index < 0x100]
    for i in xrange(0x100):
        AIFixObject.set_formation_ai(i, False)
    for n in list(new_formations):
        if n.index >= 0x100:
            u = unused_formations.pop()
            u.reassign_data(n)
            new_formations.remove(n)
            new_formations.append(u)
            n = u
        if n.is_lunar_ai_formation:
            AIFixObject.set_formation_ai(n.index, True)

    bosses = [f for f in new_formations if f.get_bit("no_flee") and
              f.battle_music != 0]
    zeromus = [b for b in bosses if zeromon1 in b.monsters
               and zeromon2 in b.monsters][0]
    nonbosses = [f for f in new_formations if f not in bosses]
    bosses.remove(zeromus)
    randoms = random.sample(nonbosses, 200)
    extras = [f for f in nonbosses if f not in randoms]

    randoms = sorted(randoms, key=lambda f: (f.rank, f.index))
    to_assign = list(randoms)
    max_pack = len(PackObject.every)-1
    packdict = defaultdict(list)

    def index_modifier():
        return random.random() * random.random() * random.choice([0.1, -0.1])

    while True:
        random.shuffle(to_assign)
        for f in to_assign:
            index = randoms.index(f) / float(len(randoms)-1)
            if index <= 0.5:
                index *= (1 + index_modifier())
            else:
                index = 1-index
                index *= (1 + index_modifier())
                index = 1-index
            index = index * max_pack
            index = int(round(min(max_pack, max(0, index))))
            if (index < max_pack and len(packdict[index]) > 1
                    and len(packdict[index+1]) <= 1):
                index += 1
            elif (index > 0 and len(packdict[index]) > 1
                    and len(packdict[index-1]) <= 1):
                index -= 1
            if len(packdict[index]) < 7:
                packdict[index].append(f)
                assert 0 < len(packdict[index]) < 8

        if all([len(packdict[i]) >= 2 for i in xrange(max_pack+1)]):
            break

    for i in xrange(max_pack+1):
        formations = packdict[i]
        if len(formations) >= 7:
            continue
        monsters = set([m for f in formations for m in f.monsters])

        necessary_monsters = set([])
        for m in sorted(monsters):
            if all([m in f.monsters for f in formations]):
                necessary_monsters.add(m)
        if necessary_monsters:
            candidates = [f for f in randoms + extras
                          if necessary_monsters <= set(f.monsters)]
        else:
            candidates = [f for f in randoms + extras
                          if set(f.monsters) <= monsters]
        while len(formations) < 7:
            formations.append(random.choice(candidates))
        for f in formations:
            if f in extras:
                extras.remove(f)
        random.shuffle(packdict[i])

    packids = range(max_pack+1)
    pack_assignments = defaultdict(set)
    for i, cg in enumerate(cluster_groups):
        portion = len(cg.mapids) / float(sum(
            [len(cg2.mapids) for cg2 in cluster_groups[i:]]))
        num_packs = int(round(portion * len(packids)))
        num_packs = max(num_packs, 1)
        for p in packids[:num_packs]:
            pack_assignments[i].add(p)
            packids.remove(p)

    formation_assignments = defaultdict(set)
    for i in pack_assignments:
        packids = pack_assignments[i]
        formations = set([f for p in packids for f in packdict[p]])
        formation_assignments[i] |= formations

    boss_assignments = []
    prev_boss_rank = 0
    for i in xrange(len(cluster_groups)-1):
        lower = max([f.rank for f in formation_assignments[i]])
        j = i
        while True:
            j += 1
            if j >= len(cluster_groups)-1:
                upper = None
                break
            upper = max([f.rank for f in formation_assignments[j]])
            boss_candidates = set([b for b in bosses if lower < b.rank])
            next_candidates = set([b for b in bosses if upper < b.rank])
            boss_candidates -= next_candidates
            if not boss_candidates:
                continue
            boss_candidates = sorted(boss_candidates,
                                     key=lambda b: (b.rank, b.index))
            temp = [b for b in boss_candidates if b.rank > prev_boss_rank]
            if temp:
                boss_candidates = temp
            max_index = len(boss_candidates)-1
            index = random.randint(0, random.randint(0, max_index))
            chosen = boss_candidates[index]
            bosses.remove(chosen)
            boss_assignments.append(chosen)
            prev_boss_rank = max(chosen.rank, prev_boss_rank)
            break
        if upper is None:
            break

    remaining = len(cluster_groups) - len(boss_assignments) - 1
    if remaining:
        boss_candidates = [b for b in bosses if b.rank > prev_boss_rank]
        if len(boss_candidates) < remaining:
            last_bosses = sorted(bosses,
                                 key=lambda b: b.rank)[-remaining:]
        else:
            last_bosses = random.sample(boss_candidates, remaining)
        boss_assignments += sorted(
            last_bosses, key=lambda b: (b.rank, b.index))
        for b in last_bosses:
            bosses.remove(b)

    assert len(boss_assignments) == len(cluster_groups)-1
    boss_assignments.append(zeromus)

    for b, cg in zip(boss_assignments, cluster_groups):
        cg.boss = b

    for i, cg in enumerate(cluster_groups):
        if bosses:
            if hasattr(cg, "boss"):
                upper = cg.boss.rank
            else:
                upper = max([c.rank for c in bosses]) + 1
            if i > 0:
                b2 = boss_assignments[i-1]
                lower = b2.rank
            else:
                lower = 0
            candidates = [c for c in bosses if lower <= c.rank < upper]
            if candidates:
                packs = [packdict[p] for p in sorted(pack_assignments[i])]
                random.shuffle(packs)
                if len(candidates)*2 < len(packs):
                    candidates = candidates + candidates
                random.shuffle(candidates)
                for c, p in zip(candidates, packs):
                    assert len(p) == 7
                    p.append(c)
        packs = [packdict[p] for p in sorted(pack_assignments[i])]
        for p in packs:
            if len(p) >= 8 and p[7] in bosses:
                bosses.remove(p[7])
            elif len(p) < 8:
                assert len(p) == 7
                p.append(random.choice(p))
        cg.packs = packs
        cg.formations = sorted(set([f for p in cg.packs for f in p]))

    assert all(hasattr(cg, "boss") for cg in cluster_groups)

    for p in PackObject.every:
        p.formation_ids = [f.index for f in packdict[p.index]]
        for f in p.formations:
            assert f.rank > 0

    for cgi, cg in enumerate(cluster_groups):
        packs = sorted(pack_assignments[cgi])
        mapids = list(cg.ranked_mapids)
        if len(packs) > len(mapids):
            packs = sorted(random.sample(packs, len(mapids)))
        max_pack = len(packs)-1
        max_map = float(len(mapids)-1)
        for i, m in enumerate(mapids):
            index = int(round((i / max_map) * max_pack))
            pack = packs[index]
            assert m < 0x100
            e = EncounterObject.get(m)
            e.encounter = pack

    for cgi, cg in enumerate(cluster_groups):
        for p in cg.packs:
            for f in p:
                if f.get_bit("no_flee"):
                    f.set_bit("no_flee", False)
                else:
                    f.set_music(3)
        if cgi >= len(cluster_groups)-1:
            assert cg.boss == zeromus
            cg.boss.set_music(3)
            # set correct background for zeromus
            f = open(get_outfile(), 'r+b')
            f.seek(addresses.zeromus_background)
            write_multi(f, cg.boss.index, length=2)
            f.seek(addresses.zeromus_background+2)
            f.write(chr(0xD0))
            f.close()
        elif cg.boss.battle_music == 3:
            if cgi > len(cluster_groups) / 2:
                cg.boss.set_music(2)
            else:
                cg.boss.set_music(1)

    zeromus.monster_types = [255, 201, 255]
    zeromus.monster_qty = 0b00010000


def generate_cave_layout(segment_lengths=None):
    LUNG_INDEX = 0xbc
    ALTERNATE_PALETTE_BATTLE_BGS = [0, 4, 7, 8, 11, 12]
    if segment_lengths is None:
        #upper limit is ~160 for now
        #segment_lengths = [10] + ([11] * 7) + [12]
        segment_lengths = [10] * 16
        segment_lengths = [3] * 16

    clusterpath = path.join(tblpath, "clusters.txt")
    bannedgridpath = path.join(tblpath, "banned_grids.txt")
    f = open(clusterpath)
    clusters = [Cluster.from_string(line) for line in f.readlines()
                if line.strip() and line[0] != '#']
    f.close()
    mapids = sorted(set([c.mapid for c in clusters]),
                    key = lambda m: (max(
                        [len(mo.all_exits) for mo in
                         MapObject.reverse_grid_index(m)]), m))
    f = open(bannedgridpath)
    banned_grids = set([int(line, 0x10) for line in f.readlines()
                        if line.strip()])

    PALETTE_BANNED = [(13,  0),]
    palette_options = defaultdict(set)
    for m in MapObject.every:
        if m.grid_index in banned_grids:
            continue
        if (m.battle_bg, m.map_palette) in PALETTE_BANNED:
            continue
        palette_options[(m.tileset_index, m.battle_bg)].add(
            (m.map_palette, m.get_bit("alternate_palette")))

    giant_lung = MapObject.reverse_grid_index_canonical(LUNG_INDEX)
    banned_grids.add(giant_lung.grid_index)
    banned_grids.add(giant_lung.background)
    for m in MapObject.every:
        if m.index >= 0x100:
            break
        if m.grid_index not in mapids:
            continue
        if m.grid_index not in banned_grids and m.grid_index != m.background:
            banned_grids.add(m.background)
    f.close()
    mapids = [m for m in mapids if m not in banned_grids]
    chosen = []
    to_replace = []
    replace_dict = {}

    LUNAR_WHALE_INDEX = 0x12c
    ZEMUS_INDEX = 0x172
    lunar_whale = MapGridObject.superget(LUNAR_WHALE_INDEX)
    lunar_whale_map = lunar_whale.map
    if get_global_label() == "FF2_US_11":
        lunar_whale_map[10][7] = 0x7F
    else:
        lunar_whale_map[10][7] = 0x7E
    lunar_whale.overwrite_map_data(lunar_whale_map)
    zemus = MapGridObject.superget(ZEMUS_INDEX)
    zemus_map = zemus.map
    zemus_map[15][15] = 0x45
    zemus_map[23][14] = 0x60
    zemus_map[23][15] = 0x45
    zemus_map[23][16] = 0x60
    zemus.overwrite_map_data(zemus_map)
    special_maps = [LUNAR_WHALE_INDEX, ZEMUS_INDEX]
    while len(chosen) < sum(segment_lengths):
        for m in special_maps:
            if m not in replace_dict:
                choose = m
                break
        else:
            max_index = len(mapids)-1
            try:
                choose = random.randint(random.randint(0, max_index), max_index)
            except ValueError:
                raise Exception("Dungeon too large; not enough maps.")
            choose = mapids[choose]
            mapids.remove(choose)
        if choose >= 0x100:
            size = MapGrid2Object.get(choose & 0xFF).size
            candidates = [m for m in MapGridObject.every
                          if m.index in mapids and m.size >= size]
            priority = [m for m in MapGridObject.every
                        if m.index not in mapids + chosen + to_replace
                        and m.size >= size and m.index not in banned_grids]
            sort_func = lambda m: (m.size, m.index)
            candidates = (sorted(priority, key=sort_func) +
                          sorted(candidates, key=sort_func))
            # the zeroth map should have trigger data that starts with a low
            # byte value or else it will be mistaken by ff4kster for another
            # pointer in the table
            candidates = [c for c in candidates
                          if MapObject.reverse_grid_index(c.index)
                          and c.index > 0]
            if not candidates:
                continue
            max_index = len(candidates)-1
            candchoose = random.randint(0, random.randint(0, max_index))
            candchoose = candidates[candchoose]
            to_replace.append(candchoose.index)
            if candchoose not in priority:
                mapids.remove(candchoose.index)
            replace_dict[choose] = candchoose.index
            assert not set(to_replace) & set(mapids)
        if choose not in special_maps:
            chosen.append(choose)
    assert not set(chosen) & set(special_maps)
    clusters = [c for c in clusters if c.mapid in chosen]

    cluster_groups = [ClusterGroup([]) for _2 in xrange(len(segment_lengths))]
    for i, cg in enumerate(cluster_groups):
        cg.index = i
    to_assign = sorted(clusters)
    random.shuffle(to_assign)
    while to_assign:
        cluster_more = []
        for i, cg in enumerate(cluster_groups):
            if segment_lengths[i] > len(cg.mapids):
                cluster_more.append(cg)
        mapid = to_assign[0].mapid
        assign_next = [c for c in to_assign if c.mapid == mapid]
        cg = random.choice(cluster_more)
        for a in assign_next:
            cg.clusters.append(a)
            to_assign.remove(a)

    num_unhealthy = len(cluster_groups)
    for _2 in xrange(200):
        num_unhealthy = len([cg for cg in cluster_groups if cg.health < 1])
        if num_unhealthy == 0:
            break
        cluster_groups = sorted(cluster_groups,
                                key=lambda cg: (cg.health, cg.index))
        unhealthy = [cg for cg in cluster_groups if cg.health < 1]
        healthy = [cg for cg in cluster_groups if cg.health > 1]
        max_index = len(unhealthy)-1
        index = random.randint(0, random.randint(0, max_index))
        cgu = unhealthy[index]
        max_index = len(healthy)-1
        if max_index < 0:
            cluster_groups = []
            break
        index = random.randint(random.randint(0, max_index), max_index)
        cgh = healthy[index]

        unhealthy = sorted(cgu.clusters, key=lambda c: (c.health, c))
        healthy = sorted(cgh.clusters, key=lambda c: (c.health, c))
        max_index = len(unhealthy)-1
        index = random.randint(0, random.randint(0, max_index))
        aa = unhealthy[index].mapid
        aa = [c for c in unhealthy if c.mapid == aa]
        max_index = len(healthy)-1
        index = random.randint(random.randint(0, max_index), max_index)
        bb = healthy[index].mapid
        bb = [c for c in healthy if c.mapid == bb]
        for a in aa:
            cgu.clusters.remove(a)
            cgh.clusters.append(a)
        for b in bb:
            cgu.clusters.append(b)
            cgh.clusters.remove(b)

    for cg in cluster_groups:
        for _2 in xrange(5):
            try:
                cg.full_execute()
                break
            except CaveException:
                continue
        else:
            cluster_groups = []
            break

    if [len(cg.mapids) for cg in cluster_groups] != segment_lengths:
        print "Retrying cave generation..."
        return generate_cave_layout(segment_lengths)

    random.shuffle(cluster_groups)

    for after, before in replace_dict.items():
        assert before in to_replace
        assert after >= 0x100
        for c in clusters:
            if c.mapid == after:
                c.mapid = before
        bb = MapObject.reverse_grid_index(before)
        b = bb[0]
        for b2 in bb:
            b2.neutralize()
        aa = MapObject.reverse_grid_index(after)
        a = MapObject.reverse_grid_index_canonical(after)
        b.reassign_data(a)
        b.grid_index = before
        for a2 in aa:
            b.acquire_triggers(a2)
            a2.neutralize()
        before = MapGridObject.get(before)
        after = MapGrid2Object.get(after & 0xFF)
        assert after.size <= before.size
        before.overwrite_map_data(after)

    active_clusters = sorted(set([c for cg in cluster_groups
                                  for c in cg.clusters]))
    active_maps = sorted(set([c.mapid for c in active_clusters]))
    for mapid in active_maps:
        assert mapid < 0x100
        aa = MapObject.reverse_grid_index(mapid)
        a = MapObject.reverse_grid_index_canonical(mapid)
        if a.grid_index in to_replace:
            companions = [c for c in MapObject.every
                          if c.grid_index not in banned_grids
                          and c.grid_index not in to_replace
                          and c.grid_index in mapids
                          and c.index <= 0x100
                          and c.tileset_index == a.tileset_index
                          and c.grid_index != c.background]
            if companions:
                companions = [c.background for c in companions]
                chosen, _ = Counter(companions).most_common()[0]
                a.background = chosen
            else:
                a.background = a.grid_index
                a.bg_properties = 0x86
        for a2 in aa:
            if a2 is not a:
                a.acquire_triggers(a2)
                a2.neutralize()
        assert len(MapObject.reverse_grid_index(mapid)) == 1
        triggers = TriggerObject.getgroup(a.index)
        for t in triggers:
            if t.is_exit:
                t.groupindex = -1

        m = MapObject.get(mapid)
        options = sorted(palette_options[(m.tileset_index, m.battle_bg)])
        if options and len(options) > 1:
            palette, truth = random.choice(options)
            m.map_palette = palette
            m.set_bit("alternate_palette", truth)

    for m in MapObject.every:
        m.name_index = 0

    bgm_candidates = [
        6, 13, 49,                              # overworld themes
        12, 23, 25, 27, 28, 30, 37, 40, 52,     # dungeon themes
        20, 32, 46, 50,                         # castle themes
        51,                                     # town themes
        #15,                                     # misc themes
        ]
    while len(cluster_groups) > len(bgm_candidates):
        bgm_candidates.append(random.choice(bgm_candidates))
    bgms = random.sample(bgm_candidates, len(cluster_groups))
    for cg, bgm in zip(cluster_groups, bgms):
        cg.music = bgm

    for i, cg in enumerate(cluster_groups):
        base_rank = sum(segment_lengths[:i])
        for c in cg.clusters:
            c.cluster_group = cg
            m = MapObject.reverse_grid_index_canonical(c.mapid)
            m.name_index = (base_rank + c.rank + 1) % 100
            m.music = cg.music
        cg.create_exit_triggers()

    lunar_whale = MapObject.reverse_grid_index_canonical(
        replace_dict[LUNAR_WHALE_INDEX])
    for m in MapObject.every:
        if m.grid_index not in active_maps and m != lunar_whale:
            for t in m.triggers:
                t.groupindex = -1

    zemus = MapObject.reverse_grid_index_canonical(replace_dict[ZEMUS_INDEX])
    zemus.background = zemus.index
    zemus.bg_properties = 0x86
    zemus.name_index = 100
    zemus.music = 11
    lunar_whale.music = 14
    zemus.set_battle_bg(0)
    EncounterRateObject.get(zemus.index).encounter_rate = 0
    EncounterRateObject.get(lunar_whale.index).encounter_rate = 0

    used_maps = set([MapObject.reverse_grid_index_canonical(m)
                     for m in active_maps])
    used_maps |= set([MapObject.get(m.background) for m in used_maps])
    used_maps.add(giant_lung)
    used_maps.add(lunar_whale)
    used_maps.add(zemus)
    used_maps.add(MapObject.get(giant_lung.background))
    unused_maps = [m for m in MapObject.every
                   if m not in used_maps and m.index < 0xF8
                   and m.grid_index not in banned_grids]
    for u in unused_maps:
        u.neutralize()
    used_events = set([e for m in MapObject.every for e in m.events])
    banned_events = set([e for e in EventObject.every if e.index in
                         [0x76, 0x77, 0x78]])
    used_events |= banned_events
    unused_events = [e for e in EventObject.every if e not in used_events]
    unused_events = sorted(unused_events, key=lambda e: (e.size, e.index))
    used_event_calls = set([e for m in MapObject.every for e in m.event_calls])
    unused_event_calls = [e for e in EventCallObject.every
                          if e not in used_event_calls]
    unused_event_calls = sorted(unused_event_calls,
                                key=lambda e: (e.size, e.index))
    used_speeches = set([e for m in MapObject.every for e in m.speeches])
    unused_speeches = [e for e in SpeechObject.every if e not in used_speeches
                       and e.index > 0]
    used_placement_indexes = set([m.npc_placement_index
                                  for m in MapObject.every])
    unused_placement_indexes = [p for p in range(0xFE)
                                if p not in used_placement_indexes]
    unused_flags = range(88, 0xFE)
    unused_flags.remove(225)

    def make_simple_event_call(event, npc=False):
        candidate_events = [e for e in unused_events if e.size >= len(event)]
        chosen = candidate_events.pop(0)
        unused_events.remove(chosen)
        chosen.overwrite_event(event)
        cases = [([], chosen.index)]
        size = len(EventCallObject.cases_to_bytecode(cases))
        candidate_event_calls = [e for e in unused_event_calls if e.size >= size]
        event_call = candidate_event_calls.pop(0)
        event_call.overwrite_event_call(cases)
        unused_event_calls.remove(event_call)
        return event_call

    start = cluster_groups[0].start
    finish = cluster_groups[-1].finish
    x = sum([a.xx for a in start]) / len(start)
    y = sum([a.yy for a in start]) / len(start)
    mapid = start[0].mapid
    TriggerObject.create_trigger(
        mapid=lunar_whale.index, x=2, y=13,
        misc1=mapid, misc2=(x|0x80), misc3=y)
    event = [
        0xF0, 0x24,
        0xF8, 0xC0,
        0xFE, zemus.index, 16, 23, 0x00,
        0xFA, zemus.music,
        0xFF,
        0xC3,
        0xFF,
        ]

    event_call = make_simple_event_call(event)
    TriggerObject.create_trigger(
        mapid=lunar_whale.index, x=12, y=13,
        misc1=0xFF, misc2=event_call.index, misc3=0x00)
    for a in start:
        TriggerObject.create_trigger(
            mapid=a.mapid, x=a.x, y=a.y,
            misc1=lunar_whale.index, misc2=(2|0x80), misc3=13)

    x = sum([a.xx for a in finish]) / len(finish)
    y = sum([a.yy for a in finish]) / len(finish)
    mapid = finish[0].mapid
    TriggerObject.create_trigger(
        mapid=zemus.index, x=14, y=23,
        misc1=mapid, misc2=(x|0x80), misc3=y)
    TriggerObject.create_trigger(
        mapid=zemus.index, x=16, y=23,
        misc1=lunar_whale.index, misc2=(12|0x80), misc3=13)
    for a in finish:
        TriggerObject.create_trigger(
            mapid=a.mapid, x=a.x, y=a.y,
            misc1=zemus.index, misc2=(14|0x80), misc3=23)

    warriors = [4, 10, 12, 9, 1, 6, 0]
    mages = [5, 11, 3]
    twins = [7, 8]
    character_actors = {
        9: 10,
        10: 13,
        11: 2,
        12: 17,
    }
    placement_index = unused_placement_indexes.pop()
    lunar_whale.npc_placement_index = placement_index
    for p in lunar_whale.npc_placements:
        p.groupindex = -1

    def create_recruitment_npc(character, x, y):
        if character in character_actors:
            actor = character_actors[character]
        else:
            actor = character
        if get_global_label() == "FF2_US_11":
            event = [0xF8, 0x7B]
        else:
            event = [0xF8, 0xC0]
        event += [
            0xE7, actor+1,
            0xFF,
            0xE8, actor+1,
            0xFF,
            ]
        candidate_events = [e for e in unused_events
                            if e.size >= len(event)]
        chosen = candidate_events.pop(0)
        unused_events.remove(chosen)
        chosen.overwrite_event(event)
        cases = [([], chosen.index)]
        size = len(SpeechObject.cases_to_bytecode(cases))
        candidate_speeches = [e for e in unused_speeches
                              if e.size >= size
                              and e.index < 0x100]
        speech = candidate_speeches.pop(0)
        speech.overwrite_event_call(cases)
        unused_speeches.remove(speech)
        NPCSpriteObject.get(speech.index).sprite = character
        NPCVisibleObject.set_visible(speech.index)
        p = PlacementObject.create_npc_placement(
            speech.index, lunar_whale.npc_placement_index, x, y)
        p.set_bit("marches", True)
        p.set_bit("turns", False)
        p.set_bit("walks", False)
        p.set_bit("intangible", False)
        assert p.facing == 2

    for i, c in enumerate(warriors):
        x = 4 + i
        y = 7
        create_recruitment_npc(c, x, y)

    for i, c in enumerate(mages):
        x = 2
        y = 8 + i
        create_recruitment_npc(c, x, y)

    for i, c in enumerate(twins):
        x = 12
        y = 8 + i
        create_recruitment_npc(c, x, y)

    cluster_groups[0].home = lunar_whale.index, 7, 10

    assign_formations(cluster_groups)

    ZEMUS_FLAME = 66
    placement_index = unused_placement_indexes.pop()
    zemus.npc_placement_index = placement_index
    for p in zemus.npc_placements:
        p.groupindex = -1

    # final boss / zeromus battle
    event = [
        0xFA, 0x07,
        0xEC, cluster_groups[-1].boss.index,
        0xE9, 0x02,
        0x08,
        0xFB, 0x55,
        0xE9, 0x04,
        0xFA, 0x03,
        0xE9, 0x20,
        0xDA,
        0xFA, 0x37,
        0xFD, 0x39,
        0xFF,
        ]
    candidate_events = [e for e in unused_events
                        if e.size >= len(event)]
    chosen = candidate_events.pop(0)
    unused_events.remove(chosen)
    chosen.overwrite_event(event)
    cases = [([], chosen.index)]
    size = len(SpeechObject.cases_to_bytecode(cases))
    candidate_speeches = [e for e in unused_speeches
                          if e.size >= size
                          and e.index < 0x100]
    speech = candidate_speeches.pop(0)
    speech.overwrite_event_call(cases)
    unused_speeches.remove(speech)
    NPCSpriteObject.get(speech.index).sprite = ZEMUS_FLAME
    NPCVisibleObject.set_visible(speech.index)
    p = PlacementObject.create_npc_placement(
        speech.index, zemus.npc_placement_index, 15, 8)
    p.set_bit("marches", True)
    p.set_bit("turns", False)
    p.set_bit("walks", False)
    p.set_bit("intangible", False)
    p.misc |= 0xC0
    assert p.facing == 2

    summons = range(54, 64)
    random.shuffle(summons)
    tellah_initial_spells = [i.spell for i in InitialSpellObject.getgroup(5) +
                             InitialSpellObject.getgroup(6)]
    tellah_learn_spells = [i for i in xrange(1, 0x31)
                           if i not in tellah_initial_spells]
    try:
        learn_increment = (len(tellah_learn_spells) /
                           float(len(cluster_groups)-1))
        if learn_increment > int(learn_increment):
            learn_increment = int(learn_increment)+1
    except ZeroDivisionError:
        learn_increment = 999

    all_battle_bgs = [(x, False) for x in range(0x10)]
    all_battle_bgs.extend([(x, True) for x in ALTERNATE_PALETTE_BATTLE_BGS])
    lung_battle_bgs = list(all_battle_bgs)
    done_battle_bgs = set([
        (MapObject.get(mapid).battle_bg,
         MapObject.get(mapid).get_bit("alternate_palette"))
        for cg in cluster_groups for mapid in cg.mapids])
    lung_battle_bgs = [l for l in lung_battle_bgs if l not in done_battle_bgs]
    assert len(cluster_groups)-1 < len(all_battle_bgs)
    while len(lung_battle_bgs) < len(cluster_groups)-1:
        b = random.choice(all_battle_bgs)
        if b in lung_battle_bgs:
            continue
        lung_battle_bgs.append(b)
    random.shuffle(lung_battle_bgs)

    LUNG_SONG = 2  # long way to go
    lungs = []
    counter = 0
    for aa, bb in zip(cluster_groups, cluster_groups[1:]):
        counter += 1
        formation = aa.boss.index
        aa = aa.finish
        bb = bb.start
        lung = unused_maps.pop()
        lung.copy_data(giant_lung)
        lung.npc_placement_index = PlacementObject.canonical_zero
        lung.name_index = 100
        lung.music = LUNG_SONG
        bbg, truth = lung_battle_bgs.pop()
        lung.set_battle_bg(bbg)
        lung.set_bit("alternate_palette", truth)
        EncounterRateObject.get(lung.index).encounter_rate = 0

        ax = sum([a.xx for a in aa]) / len(aa)
        ay = sum([a.yy for a in aa]) / len(aa)
        amap = aa[0].mapid
        t = TriggerObject.create_trigger(
            mapid=lung.index, x=15, y=24,
            misc1=MapObject.reverse_grid_index_canonical(amap).index,
            misc2=(ax|0x80), misc3=ay)
        for a in aa:
            a.create_exit_trigger(lung.index, t.x, t.y, use_given_mapid=True)

        bx = sum([b.xx for b in bb]) / len(bb)
        by = sum([b.yy for b in bb]) / len(bb)
        bmap = bb[0].mapid
        t = TriggerObject.create_trigger(
            mapid=lung.index, x=15, y=04,
            misc1=MapObject.reverse_grid_index_canonical(bmap).index,
            misc2=(bx|0x80), misc3=by)
        for b in bb:
            b.create_exit_trigger(lung.index, t.x, t.y, use_given_mapid=True)

        flag = unused_flags.pop()
        boss_event = [
            #0xD8,
            #0xE9, 0x08,
            0xFB, 0x47,
            0xFD, 0x07,
            0xFB, 0x00,
            0xEC, formation,
            0xF2, flag,
            0xFA, LUNG_SONG,
            #0xFF,
            ]

        if get_global_label() == "FF2_US_11":
            yesno_boss_event = [0xF8, 0x98] + boss_event + [0xFF, 0xFF]
        else:
            yesno_boss_event = [0xF8, 0x98, 0xFF, 0xFF] + boss_event + [0xFF]

        if summons:
            summon = summons.pop()
            boss_event += [0xE2, 0x04, summon]
        if learn_increment >= len(tellah_learn_spells):
            to_learn = list(tellah_learn_spells)
        else:
            to_learn = random.sample(tellah_learn_spells, learn_increment)
        for spell in to_learn:
            if spell >= 0x19:
                boss_event += [0xE2, 0x06, spell]
            else:
                boss_event += [0xE2, 0x05, spell]
            tellah_learn_spells.remove(spell)
        boss_event += [0xFF]

        candidate_events = [e for e in unused_events
                            if e.size >= len(boss_event)]
        boss_chosen = candidate_events.pop(0)
        unused_events.remove(boss_chosen)
        candidate_events = [e for e in unused_events
                            if e.size >= len(yesno_boss_event)]
        yesno_boss_chosen = candidate_events.pop(0)
        unused_events.remove(yesno_boss_chosen)
        boss_chosen.overwrite_event(boss_event)
        yesno_boss_chosen.overwrite_event(yesno_boss_event)
        cases = [([(flag, False)], boss_chosen.index),
                 ([], yesno_boss_chosen.index)]
        size = len(EventCallObject.cases_to_bytecode(cases))
        candidate_event_calls = [e for e in unused_event_calls
                                 if e.size >= size]
        event_call = candidate_event_calls.pop(0)
        event_call.overwrite_event_call(cases)
        unused_event_calls.remove(event_call)
        t = TriggerObject.create_trigger(
            mapid=lung.index, x=15, y=15,
            misc1=0xFF, misc2=event_call.index, misc3=0)

    return cluster_groups, lungs


class StatusLoadObject(TableObject): pass
class StatusSaveObject(TableObject): pass


class AIFixObject(TableObject):
    @staticmethod
    def set_formation_ai(index, truth):
        index, bit = index / 8, index % 8
        bit = 1 << bit
        a = AIFixObject.get(index)
        if truth:
            a.formation_flags |= bit
        elif a.formation_flags & bit:
            a.formation_flags ^= bit


class FormationObject(TableObject):
    def __repr__(self):
        return "%x %s" % (self.index, self.signature)

    def reassign_data(self, other):
        if self is other:
            raise Exception("Both formations are the same.")
        self.copy_data(other)
        self._rank = other._rank
        self._is_lunar_ai_formation = other.is_lunar_ai_formation

    @property
    def is_lunar_ai_formation(self):
        if hasattr(self, "_is_lunar_ai_formation"):
            return self._is_lunar_ai_formation
        for m in MapObject.every:
            if m.is_lunar_ai_map:
                if EncounterObject.get(m.index).encounter == 0:
                    if self in m.event_battles:
                        self._is_lunar_ai_formation = True
                        break
                elif self in m.all_formations:
                    self._is_lunar_ai_formation = True
                    break
        else:
            self._is_lunar_ai_formation = False
        return self.is_lunar_ai_formation

    @property
    def monsters(self):
        monsters = []
        for i, t in enumerate(self.monster_types):
            try:
                m = MonsterObject.get(t)
            except KeyError:
                continue
            num = (self.monster_qty >> (6 - (2*i))) & 0b11
            monsters.extend([m for _ in xrange(num)])
        return monsters

    @property
    def signature(self):
        monster_names = [m.name for m in self.monsters]
        s = []
        for name, count in sorted(Counter(monster_names).items()):
            s.append("%s x%s" % (name, count))
        return ", ".join(s)

    @property
    def is_boss(self):
        return self.get_bit("no_flee") and self.battle_music != 0

    @property
    def sum_rank(self):
        return sum(m.rank for m in self.monsters)

    @property
    def weighted_xp_rank(self):
        rank = 0
        monsters = sorted(self.monsters,
                          key=lambda m: m.rank, reverse=True)
        assert monsters[0].rank >= monsters[-1].rank
        for i, m in enumerate(monsters):
            rank += (m.rank * (0.5 ** i))
        return rank

    @property
    def map_rank(self):
        if self.index == 0xe5:
            maps = [MapObject.get(0x2c)]
        else:
            maps = [m for m in MapObject.every if self in m.all_formations]
        if not maps:
            return -1
        if len(maps) == 1:
            m = maps[0]
            m2s = set([
                m4 for m2 in m.connected_maps
                for m3 in m2.connected_maps
                for m4 in m3.connected_maps] + [m])
            if m.index != 0x6C:
                m2s = [m for m in m2s
                       if EncounterObject.get(m.index).encounter != 0]
            formations = [f for m2 in m2s for f in m2.formations]
            if formations:
                return max(f._rank for f in formations) + 1
            return None
        return None

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank + (self.index / 1000.0)
        f = open(path.join(tblpath, "formation_ranks.txt"))
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            rank, index, etc = line.split(" ", 2)
            rank = int(rank)
            index = int(index, 0x10)
            fo = FormationObject.get(index)
            assert not hasattr(fo, "_rank")
            fo._rank = rank
        f.close()
        for f in FormationObject.every:
            if not hasattr(f, "_rank"):
                f._rank = -1
        return self.rank

    @property
    def battle_music(self):
        return (self.misc2 >> 2) & 0b11

    def set_music(self, value):
        self.misc2 = self.misc2 & 0b11110011
        self.misc2 |= (value << 2)
        assert self.battle_music == value

    @property
    def scenario(self):
        return (self.get_bit("no_gameover") or
                self.get_bit("character_battle") or
                self.get_bit("auto_battle"))

    @staticmethod
    def rank_properly():
        for f in FormationObject.every:
            if not f.get_bit("no_flee"):
                f._rank = f.weighted_xp_rank
            elif f.weighted_xp_rank == 0 or f.scenario:
                f._rank = -1

        for f in FormationObject.every:
            if hasattr(f, "_rank"):
                continue
            monsters = set(f.monsters)
            for f2 in FormationObject.every:
                if not hasattr(f2, "_rank") or f2.rank <= 0:
                    continue
                if (f.weighted_xp_rank == f2.weighted_xp_rank
                        and monsters == set(f2.monsters)):
                    f._rank = f2._rank

        for f in FormationObject.every:
            if hasattr(f, "_rank"):
                continue
            monsters = set(f.monsters)
            f2s = [f2 for f2 in FormationObject.every
                   if hasattr(f2, "_rank") and f2.rank > 0
                   and set(f2.monsters) & monsters]
            f2s = sorted(f2s, key=lambda f2: abs(
                f2.weighted_xp_rank - f.weighted_xp_rank))
            if f2s:
                f2 = f2s[0]
                f._rank = f2._rank
                w1, w2 = f.weighted_xp_rank, f2.weighted_xp_rank
                if w1 > w2:
                    f._rank += 1
                elif w2 > w1:
                    f._rank -= 1

        for f in FormationObject.every:
            if hasattr(f, "_rank"):
                continue
            map_rank = f.map_rank
            if map_rank is not None:
                f._rank = map_rank

        for f in FormationObject.every:
            if hasattr(f, "_rank"):
                continue
            f._rank = f.weighted_xp_rank

        return


class NameObject(TableObject):
    @property
    def name(self):
        s = ""
        for c in self.text:
            if 66 <= c <= 91:
                s += chr(c-1)
            elif 92 <= c <= 117:
                s += chr(c+5)
            elif 128 <= c <= 137:
                s += "%s" % (c-128)
            elif c == 255:
                s += " "
            else:
                s += "?"
        return s.rstrip(' ')

    def rename(self, name):
        s = ""
        if isinstance(name, basestring):
            name = [name]
        for n in name:
            if n and n[0] == '$':
                s += chr(int(n[1:], 0x10))
            else:
                s += text_to_bytestr(n)
        self.text = map(ord, s)
        while len(self.text) < self.total_size:
            self.text += [ord(text_to_bytestr(' '))]
        assert len(self.text) == self.total_size


class MonsterNameObject(NameObject): pass


class MonsterDropObject(TableObject):
    @property
    def pretty_items(self):
        s = ""
        for d in self.items:
            s += ItemNameObject.get(d).name + "\n"
        return s.strip()


class MonsterGilObject(TableObject): pass
class MonsterXPObject(TableObject): pass


class MonsterObject(TableObject):
    minmaxes = {}

    def __repr__(self):
        return "%s %s %s" % ("{0:0>2}".format("%x" % self.index), int(round(self.rank)), self.name)

    @classmethod
    def write_all(cls, filename):
        pass

    @property
    def name(self):
        if get_global_label() == "FF2_US_11":
            return MonsterNameObject.get(self.index).name
        else:
            return "MONSTER_%x" % self.index

    def get_attr_rank(self, attr):
        if attr not in MonsterObject.minmaxes:
            values = sorted(set([getattr(m, attr) for m in self.every]))
            MonsterObject.minmaxes[attr] = values[0], values[-1]
        low, high = MonsterObject.minmaxes[attr]
        value = getattr(self, attr)
        value = min(high, max(low, value))
        value = (value - low) / float(high - low)
        return value

    @property
    def rank(self):
        return self.xp

    @property
    def xp(self):
        return MonsterXPObject.get(self.index).xp

    @property
    def gil(self):
        return MonsterGilObject.get(self.index).gil

    @property
    def drops(self):
        return MonsterDropObject.get(self.drop_index & 0x3F)

    @property
    def pretty_drops(self):
        return self.drops.pretty_items

    @property
    def drop_rate(self):
        return self.drop_index >> 6


class EncounterRateObject(TableObject): pass


class EncounterObject(TableObject):
    def __repr__(self):
        s = "ENCOUNTER %s %s\n" % (self.index, MapObject.get(self.index).name)
        for f in self.formations:
            s += str(f) + "\n"
        return s.strip()

    @property
    def formations(self):
        return PackObject.get(self.encounter).get_formations(
            alternate=self.index >= 0x100)


class PackObject(TableObject):
    @property
    def __repr__(self):
        s = "PACK %x\n" % self.index
        for f in self.get_formations():
            s += str(f) + "\n"
        return s.strip()

    @property
    def formations(self):
        return self.get_formations()

    @property
    def rank(self):
        return (max([f.rank for f in self.formations[:7]]), self.index)

    def get_formations(self, alternate=False):
        if alternate:
            return [FormationObject.get(f+0x100) for f in self.formation_ids]
        return [FormationObject.get(f) for f in self.formation_ids]


class ItemNameObject(NameObject): pass
class SpellNameObject(NameObject): pass
class LongSpellNameObject(NameObject): pass


class PriceObject(TableObject):
    @property
    def name(self):
        return ItemNameObject.get(self.index).name

    @property
    def is_equipment(self):
        return self.index <= 175

    @property
    def is_arrows(self):
        return 0x84 <= self.index <= 0x95

    @property
    def price(self):
        if self.price_byte & 0x80:
            return (self.price_byte & 0x7F) * 1000
        return self.price_byte * 10

    @property
    def banned(self):
        return self.index in [0, 96] + range(0xFC, 0x100)

    @property
    def buyable(self):
        for s in ShopObject.every:
            if self.index < 0xFF and self.index in s.items:
                return True
        return False

    @property
    def sellable(self):
        return self.price != 0

    @property
    def rank(self):
        if self.banned:
            return -1
        if self.buyable:
            return self.price
        return 1000000


class CommandNameObject(NameObject): pass
class CharacterObject(TableObject): pass
class InitialEquipObject(TableObject): pass


class LearnedSpellObject(TableObject):
    @classproperty
    def every(self):
        if not hasattr(LearnedSpellObject, "extras"):
            LearnedSpellObject.extras = []
        return (super(LearnedSpellObject, self).every +
                LearnedSpellObject.extras)

    @staticmethod
    def add_spell(spell, level, groupindex):
        for i in LearnedSpellObject.getgroup(groupindex):
            if i.spell == spell:
                return
        index = len(LearnedSpellObject.every)
        i = LearnedSpellObject(index=index)
        i.groupindex = groupindex
        i.spell = spell
        i.level = level
        LearnedSpellObject.extras.append(i)

    @classmethod
    def groupsort(cls, objs):
        assert len(set([o.spell for o in objs])) == len(objs)
        try:
            assert len(objs) < 24
        except AssertionError:
            raise Exception("Learnset %s overload." % objs[0].groupindex)
        return sorted(objs, key=lambda o: (o.level, o.spell, o.index))


class InitialSpellObject(TableObject):
    @classproperty
    def every(self):
        if not hasattr(InitialSpellObject, "extras"):
            InitialSpellObject.extras = []
        return super(InitialSpellObject, self).every + InitialSpellObject.extras

    @staticmethod
    def add_spell(spell, groupindex):
        for i in InitialSpellObject.getgroup(groupindex):
            if i.spell == spell:
                return
        index = len(InitialSpellObject.every)
        i = InitialSpellObject(index=index)
        i.groupindex = groupindex
        i.spell = spell
        InitialSpellObject.extras.append(i)

    @classmethod
    def groupsort(cls, objs):
        assert len(set([o.spell for o in objs])) == len(objs)
        try:
            assert len(objs) < 24
        except AssertionError:
            raise Exception("Spellset %s overload." % objs[0].groupindex)
        return sorted(objs, key=lambda o: (o.spell, o.index))


class EventObject(TableObject):
    @property
    def full_event_pointer(self):
        return addresses.event_base + self.event_pointer

    @property
    def instructions(self):
        if hasattr(self, "_instructions"):
            return self._instructions
        f = open(get_outfile(), "r+b")
        f.seek(self.full_event_pointer)
        self._instructions = []
        while True:
            cmd = ord(f.read(1))
            if 0 <= cmd <= 0xDA or cmd == 0xFF:
                parameters = []
            elif cmd in [0xE2, 0xEB]:
                parameters = [ord(c) for c in f.read(2)]
            elif cmd == 0xFC:
                raise NotImplementedError
            elif cmd == 0xFE:
                parameters = [ord(c) for c in f.read(4)]
            else:
                parameters = [ord(f.read(1))]

            self._instructions.append((cmd, parameters))
            if cmd == 0xFF:
                break
        f.close()
        return self.instructions

    @property
    def pretty_script(self):
        s = "EVENT %x\n" % self.index
        for cmd, parameters in self.instructions:
            s += "{0:0>2}: ".format("%x" % cmd)
            s += " ".join(["{0:0>2}".format("%x" % p) for p in parameters])
            s += "\n"
        return s.strip().upper()

    @property
    def size(self):
        size = 0
        for cmd, parameters in self.instructions:
            size += 1 + len(parameters)
        return size

    @property
    def commands(self):
        return [cmd for (cmd, parameters) in self.instructions]

    @property
    def messager(self):
        return bool(set(self.commands) &
                    set([0xF1, 0xF0, 0xEF, 0xEE, 0xF8, 0xF6]))

    @property
    def battler(self):
        return 0xEC in self.commands

    @property
    def flagger(self):
        return bool(set(self.commands) & set([0xF2, 0xF3]))

    @property
    def map_loader(self):
        return 0xFE in self.commands

    @property
    def gfx_changer(self):
        return 0xDD in self.commands

    def overwrite_event(self, data):
        f = open(get_outfile(), "r+b")
        f.seek(self.full_event_pointer)
        f.write("".join(map(chr, data)))
        f.close()
        if hasattr(self, "_instructions"):
            delattr(self, "_instructions")


class NPCSpriteObject(TableObject): pass


class NPCVisibleObject(TableObject):
    @staticmethod
    def set_visible(index, truth=True):
        my_index = index / 8
        bit = 1 << (index % 8)
        nv = NPCVisibleObject.get(my_index)
        if truth:
            nv.visible |= bit
        elif nv.visible & bit:
            nv.visible ^= bit

    @staticmethod
    def set_invisible(index):
        NPCVisibleObject.set_visible(index, False)

    @staticmethod
    def get_visible(index):
        my_index = index / 8
        bit = 1 << (index % 8)
        nv = NPCVisibleObject.get(my_index)
        return bool(nv.visible & bit)


class EventCallObject(TableObject):
    @property
    def size(self):
        try:
            return (self.get(self.index+1).event_call_pointer
                - self.event_call_pointer)
        except KeyError:
            return 0

    @property
    def full_event_call_pointer(self):
        return addresses.eventcall_base + self.event_call_pointer

    @property
    def cases(self):
        if hasattr(self, "_cases"):
            return self._cases
        self._cases = []

        f = open(get_outfile(), "r+b")
        f.seek(self.full_event_call_pointer)
        conditions = []
        while True:
            if f.tell() >= self.full_event_call_pointer + self.size:
                break
            peek = ord(f.read(1))
            if peek == 0xFF:
                call = ord(f.read(1))
                self._cases.append((conditions, call))
                conditions = []
            elif peek == 0xFE:
                conditions.append((ord(f.read(1)), True))
            else:
                conditions.append((peek, False))
        f.close()
        return self.cases

    @property
    def events(self):
        events = []
        for conditions, call in self.cases:
            events.append(EventObject.get(call))
        return events

    @property
    def crash_game(self):
        crash_game = [0xFE, 0xFF]
        for conditions, call in self.cases:
            if call in crash_game:
                return True
        return False

    @staticmethod
    def cases_to_bytecode(cases):
        s = ""
        for conditions, call in cases:
            for flag, truth in conditions:
                assert flag <= 0xFD
                if truth:
                    s += chr(0xFE)
                s += chr(flag)
            s += chr(0xFF)
            s += chr(call)
        return s

    def overwrite_event_call(self, cases):
        f = open(get_outfile(), "r+b")
        f.seek(self.full_event_call_pointer)
        f.write(EventCallObject.cases_to_bytecode(cases))
        f.close()


class PlacementObject(TableObject):
    PLACEMENT_LIST = []

    def __init__(self, *args, **kwargs):
        super(PlacementObject, self).__init__(*args, **kwargs)
        if self.filename is None:
            PlacementObject.PLACEMENT_LIST.append(self)

    @classproperty
    def every(self):
        return super(PlacementObject, self).every + PlacementObject.PLACEMENT_LIST

    def cleanup(self):
        assert not self.npc_index == 0

    def neutralize(self):
        self.set_bit("intangible", True)
        self.set_bit("walks", False)
        self.npc_index = 5

    @staticmethod
    def create_npc_placement(speech_index, placement_index, x, y):
        existing = len(PlacementObject.getgroup(placement_index))
        index = (1 << 16) | (placement_index << 8) | existing
        p = PlacementObject(index=index)
        p.groupindex = placement_index
        p.npc_index = speech_index
        p.xmisc = x & 0x1F
        p.ymisc = y & 0x1F
        p.misc = 2
        existing = len(PlacementObject.getgroup(placement_index))
        return p

    @classproperty
    def available_placements(self):
        indexes = [i for i in xrange(0x100)
                   if len(PlacementObject.getgroup(i)) == 0]

        if (hasattr(PlacementObject, "canonical_zero")
               and PlacementObject.canonical_zero in indexes):
           indexes.remove(PlacementObject.canonical_zero)
        return indexes

    @property
    def x(self):
        return self.xmisc & 0x1F

    @property
    def y(self):
        return self.ymisc & 0x1F

    @property
    def facing(self):
        return self.misc & 0b11

    @property
    def palette(self):
        return (self.misc >> 2) & 0b11

    @property
    def turns(self):
        return bool(self.misc & 0x10)

    @property
    def marches(self):
        return bool(self.misc & 0x20)

    @property
    def speed(self):
        return self.misc >> 6

    @property
    def speech(self):
        return self.get_speech(alternate=False)

    def get_speech(self, alternate=False):
        if alternate:
            index = self.npc_index | 0x100
        else:
            index = self.npc_index
        return SpeechObject.get(index)

    @property
    def events(self):
        return self.speech.events

    @property
    def crash_game(self):
        return self.speech.crash_game

    @property
    def messager(self):
        return any([e.messager for e in self.events])

    @property
    def blocker(self):
        if self.get_bit("walks"):
            return False
        if not self.events:
            return True
        for e in self.events:
            if not set(e.commands) - set([0xF1, 0xF0, 0xEF, 0xEE, 0xF6, 0xFF]):
                return True
        return False


class SpeechObject(EventCallObject):
    @property
    def full_event_call_pointer(self):
        return addresses.speech_base + self.event_call_pointer


class ShopObject(TableObject): pass
class CommandObject(TableObject): pass
class MenuCommandObject(TableObject): pass
class TileObject(TableObject): pass


class TriggerObject(TableObject):
    # x, y are not modifiable without changing the map data
    # you just end up on the moon
    TRIGGER_LIST = []

    def __init__(self, *args, **kwargs):
        super(TriggerObject, self).__init__(*args, **kwargs)
        if self.filename is None:
            TriggerObject.TRIGGER_LIST.append(self)

    @classmethod
    def groupsort(cls, objs):
        assert len(set([o.groupindex for o in objs])) <= 1
        objs = sorted(objs, key=lambda o: o.index)
        treasure = [o for o in objs if o.is_chest]
        teleport = [o for o in objs if o.is_exit]
        event = [o for o in objs if o.is_event]
        return treasure + teleport + event

    @classproperty
    def every(self):
        return super(TriggerObject, self).every + TriggerObject.TRIGGER_LIST

    @staticmethod
    def create_trigger(mapid, x, y, misc1, misc2, misc3):
        existing = len(TriggerObject.getgroup(mapid))
        index = (1 << 16) | (mapid << 8) | existing
        t = TriggerObject(index=index)
        t.groupindex = mapid
        t.x = x
        t.y = y
        t.misc1 = misc1
        t.misc2 = misc2
        t.misc3 = misc3
        return t

    def neutralize(self):
        if self.is_event:
            self.misc2 = 2

    @property
    def description(self):
        s = "{0:0>2} {1:0>2}".format(self.x, self.y)
        if self.is_event:
            s = "%s EVENT: $%x" % (s, self.event_call_index)
        elif self.is_chest:
            s = "%s CHEST:" % s
            if not self.misc2 & 0xC0:
                if self.contents & 0x80:
                    s = "%s %s Gil" % (s, (self.contents & 0x7F) * 1000)
                else:
                    s = "%s %s Gil" % (s, self.contents * 10)
            if self.misc2 & 0x40:
                s = "%s %s -" % (s, FormationObject.get(self.formation))
            if self.misc2 & 0x80:
                s = "%s %s" % (s, ItemNameObject.get(self.contents).name)
        elif self.is_exit:
            s = "%s EXIT: $%x %s %s" % (s, self.new_map,
                                       self.dest_x, self.dest_y)
        return s.strip()

    @property
    def is_event(self):
        return bool(self.misc1 == 0xFF)

    @property
    def is_chest(self):
        return bool(self.misc1 == 0xFE)

    @property
    def is_exit(self):
        return not (self.is_event or self.is_chest)

    @property
    def formation(self):
        if self.groupindex < 0x100:
            return (self.misc2 & 0x3F) + 448
        else:
            return (self.misc2 & 0x3F) + 448 + 32

    @property
    def contents(self):
        return self.misc3

    @property
    def event_call_index(self):
        return self.misc2

    @property
    def event_call(self):
        return EventCallObject.get(self.event_call_index)

    @property
    def events(self):
        return self.event_call.events

    @property
    def crash_game(self):
        return self.event_call.crash_game

    @property
    def new_map(self):
        return self.misc1

    @property
    def facing(self):
        return self.misc2 >> 6

    @property
    def dest_x(self):
        return self.misc2 & 0x3F

    @property
    def dest_y(self):
        return self.misc3


class MapObject(TableObject):
    def __repr__(self):
        s = self.name + "\n"
        for i, t in enumerate(self.triggers):
            s += "{0:0>2}. ".format(i) + t.description + "\n"
        try:
            for i, (x, y) in enumerate(self.warps):
                s += "WARP {0:0>2}: {1:0>2} {2:0>2}\n".format(i, x, y)
        except KeyError:
            s += "WARP ?????\n"
        return s.strip()

    @property
    def encounter_rate(self):
        return EncounterRateObject.get(self.index).encounter_rate

    @property
    def is_lunar_ai_map(self):
        return (0x15a <= self.index <= 0x15c or
                0x167 <= self.index <= 0x17e)

    @staticmethod
    def reverse_grid_index(grid_index):
        return [m for m in MapObject.every
                if m.grid_index == grid_index & 0xFF and
                m.index & 0x100 == grid_index & 0x100 and
                m.grid_index & 0xFF < 0xFE and
                m.index & 0xFF < 0xED]

    @staticmethod
    def reverse_grid_index_canonical(grid_index):
        if not hasattr(MapObject, "reverse_canonical_dict"):
            MapObject.reverse_canonical_dict = {}
        if grid_index in MapObject.reverse_canonical_dict:
            return MapObject.reverse_canonical_dict[grid_index]
        candidates = MapObject.reverse_grid_index(grid_index)
        temp = [c for c in candidates if c.npc_placements]
        if temp:
            if len(temp) == 1:
                chosen = temp[0]
            else:
                chosen = random.choice(temp)
        else:
            chosen = candidates[0]
        MapObject.reverse_canonical_dict[grid_index] = chosen
        return MapObject.reverse_grid_index_canonical(grid_index)

    @property
    def name(self):
        #if self.name_index >= len(LOCATION_NAMES):
        #    return "?????"
        try:
            name = LOCATION_NAMES[self.name_index & 0x7F]
        except IndexError:
            name = "???????"
        return "%x %s" % (self.index, name)

    @property
    def mapgrid(self):
        if self.index < 0x100:
            return MapGridObject.get(self.grid_index)
        else:
            return MapGrid2Object.get(self.grid_index)

    @property
    def map(self):
        return self.mapgrid.map

    def pretty_tile_map(self, attr=None, validator=None):
        if validator is None and attr is not None:
            if hasattr(TileObject, attr):
                validator = lambda t: getattr(t, attr)
            else:
                validator = lambda t: t.get_bit(attr)
        s = ""
        for row in self.map:
            for tile in row:
                tile = TileObject.get((128*self.tileset_index)+tile)
                s += "#" if validator(tile) else "."
            s += "\n"
        return "%s\n%s" % (self.name, s.strip())

    @property
    def walkable_map(self):
        s = ""
        for row in self.map:
            for tile in row:
                tile = TileObject.get((128*self.tileset_index)+tile)
                if tile.get_bit("bridge_layer"):
                    s += "X"
                elif tile.get_bit("warp"):
                    s += "W"
                elif tile.get_bit("triggerable"):
                    s += "T"
                elif tile.walkable:
                    s += "."
                else:
                    s += "#"
            s += "\n"
        return "%s\n%s" % (self.name, s.strip())

    @property
    def warps(self):
        warps = []
        for y in xrange(32):
            for x in xrange(32):
                tile = self.map[y][x]
                tile = TileObject.get((128*self.tileset_index)+tile)
                if tile.get_bit("warp"):
                    warps.append((x, y))
        return warps

    @property
    def triggers(self):
        return TriggerObject.getgroup(self.index)

    @property
    def exits(self):
        return [x for x in self.triggers if x.is_exit]

    @property
    def all_exits(self):
        return self.exits + self.warps

    @property
    def chests(self):
        return [c for c in self.triggers if c.is_chest]

    @property
    def event_triggers(self):
        return [e for e in self.triggers if e.is_event]

    @property
    def events(self):
        return ([e for t in self.event_calls for e in t.events] +
                [e for p in self.speeches for e in p.events])

    @property
    def connected_maps(self):
        maps = [m for m in MapObject.every
                if self.index & 0x100 == m.index & 0x100
                and self in m.enterable_maps]
        return set(maps) | self.enterable_maps

    @property
    def enterable_maps(self):
        maps = []
        for t in self.exits:
            index = t.new_map
            if index >= 0xED:
                continue
            if self.index >= 0x100:
                index |= 0x100
            maps.append(MapObject.get(index))
        return set(maps)

    @property
    def event_battles(self):
        formations = []
        for e in self.events:
            for command, parameters in e.instructions:
                if command == 0xEC:
                    index = parameters[0]
                    if self.index >= 0x100:
                        index |= 0x100
                    formations.append(FormationObject.get(index))
        return formations

    @property
    def chest_battles(self):
        formations = []
        for c in self.chests:
            if c.misc2 & 0x40:
                f = FormationObject.get(c.formation)
                formations.append(f)
        return formations

    @property
    def formations(self):
        return EncounterObject.get(self.index).formations

    @property
    def all_formations(self):
        return self.formations + self.event_battles + self.chest_battles

    @property
    def event_calls(self):
        return [t.event_call for t in self.event_triggers]

    @property
    def speeches(self):
        alternate = self.index >= 0x100
        return [p.get_speech(alternate=alternate) for p in self.npc_placements]

    @property
    def npc_placements(self):
        if self.index >= 0x100:
            return PlacementObject.getgroup(self.npc_placement_index | 0x100)
        else:
            return PlacementObject.getgroup(self.npc_placement_index)

    @property
    def exit_summary(self):
        summary = set([])
        for e in self.exits:
            summary.add((e.x, e.y))
        for x, y in self.warps:
            summary.add((x, y))
        return summary

    @property
    def battle_bg(self):
        return self.properties_battlebackground & 0xF

    def set_battle_bg(self, value):
        self.properties_battlebackground = (
            (self.properties_battlebackground & 0xF0) | value)

    def neutralize(self):
        self.grid_index = 0xFF
        self.npc_placement_index = PlacementObject.canonical_zero
        for t in self.triggers:
            t.groupindex = -1

    def reassign_data(self, other):
        if self is other:
            raise Exception("Both maps are the same.")
        self.neutralize()
        self.copy_data(other)
        for p in other.npc_placements:
            p.groupindex = -1
        self.npc_placement_index = PlacementObject.canonical_zero
        self.acquire_triggers(other)

    def acquire_triggers(self, other):
        if self is other:
            raise Exception("Both maps are the same.")
        for t in other.triggers:
            t.groupindex = self.index


class MapGridObject(TableObject):
    @property
    def full_map_pointer(self):
        return self.map_pointer + addresses.mapgrid_base

    @property
    def map(self):
        if hasattr(self, "_map"):
            return self._map

        self._compressed = ""
        f = open(get_outfile(), "r+b")
        f.seek(self.full_map_pointer)
        data = []
        while len(data) < 1024:
            tile = ord(f.read(1))
            self._compressed += chr(tile)
            if tile & 0x80:
                tile = tile & 0x7F
                data.append(tile)
                additional = ord(f.read(1))
                self._compressed += chr(additional)
                data.extend([tile] * additional)
            else:
                data.append(tile)
        f.close()
        self._map = [data[(i*32):((i+1)*32)] for i in xrange(32)]
        assert [len(row) == 32 for row in self._map]
        assert len(self._map) == 32
        return self.map

    @property
    def compressed(self):
        self.map
        return self._compressed

    @property
    def size(self):
        return len(self.compressed)

    @property
    def pretty_map(self):
        s = ""
        for row in self.map:
            s += " ".join(["{0:0>2}".format("%x" % v) for v in row])
            s += "\n"
        return s.strip()

    @staticmethod
    def superget(index):
        if index < 0x100:
            return MapGridObject.get(index)
        else:
            return MapGrid2Object.get(index & 0xFF)

    @staticmethod
    def recompress(data):
        data = [tile for row in data for tile in row]
        previous = None
        compressed = ""
        runlength = 0
        for tile in data + [None]:
            if tile == previous:
                runlength += 1
                if runlength == 255:
                    compressed += chr(previous | 0x80) + chr(0xFF)
                    runlength = 0
            elif previous is not None:
                if runlength > 0:
                    compressed += chr(previous | 0x80) + chr(runlength)
                    runlength = 0
                else:
                    compressed += chr(previous)
            previous = tile
        return compressed

    def overwrite_map_data(self, data):
        if isinstance(data, MapGridObject):
            data = data.compressed
        else:
            data = MapGridObject.recompress(data)
        assert len(data) <= self.size
        f = open(get_outfile(), "r+b")
        f.seek(self.full_map_pointer)
        f.write(data)
        f.close()
        if hasattr(self, "_map"):
            delattr(self, "_map")


class MapGrid2Object(MapGridObject):
    @property
    def full_map_pointer(self):
        return self.map_pointer + addresses.mapgrid2_base


def setup_opening_event(mapid=0, x=16, y=30):
    chosen = random.choice(range(0, 9) + [10, 13, 17])
    new_event = [
        0xFA, 0x0E,                 # play lunar whale theme
        0xE8, 0x01,                 # remove DK cecil
        0xE7, chosen+1,             # random starting character
        0xE3, 0x00,                 # remove all statuses
        0xDE, 0xFE,                 # restore HP
        0xDF, 0xFE,                 # restore MP
        0xFE, mapid, x, y, 0x00,    # load map 0 16,30
        0xC6,
        0xFF,
        ]
    e = EventObject.get(0x10)
    e.overwrite_event(new_event)
    child_rydia = CharacterObject.get(0x2)
    child_rydia.sprite = 11


def setup_cave():
    # 9f338 - staff roll (japanese)
    f = open(get_outfile(), "r+b")
    f.seek(addresses.window_palette)
    write_multi(f, 0xc00, length=2)  # game window palette
    rng = range(0x100)
    random.shuffle(rng)
    f.seek(addresses.rng_table)
    f.write("".join(map(chr, rng)))  # new rng table
    f.close()
    npc_whitelist = set([])
    npcwpath = path.join(tblpath, "npc_whitelist.txt")
    for line in open(npcwpath).readlines():
        line = line.strip()
        if not line:
            continue
        pid, nid = line.split()
        pid = int(pid, 0x10)
        if nid == '*':
            npc = [p for p in PlacementObject.every if p.groupindex == pid]
        else:
            nid = int(nid, 0x10)
            npc = [p for p in PlacementObject.every
                   if p.groupindex == pid and p.npc_index == nid]
        npc_whitelist |= set(npc)
    for t in TriggerObject.every:
        assert not t.crash_game
    for t in TriggerObject.every + PlacementObject.every:
        if t in npc_whitelist:
            continue
        for e in t.events:
            if e.flagger or e.map_loader or e.battler or e.gfx_changer:
                t.neutralize()
                break
        else:
            if isinstance(t, PlacementObject) and (t.crash_game or t.blocker):
                t.set_bit("intangible", True)
    tblkpath = path.join(tblpath, "trigger_blacklist.txt")
    for line in open(tblkpath):
        line = line.strip()
        if not line:
            continue
        mapid, x, y = line.split()
        mapid = int(mapid, 0x10)
        x, y = int(x), int(y)
        for t in TriggerObject.every:
            if t.groupindex == mapid and t.x == x and t.y == y and t.is_event:
                t.neutralize()
    for t in TileObject.every:
        if t.get_bit("warp"):
            t.set_bit("warp", False)
            t.set_bit("triggerable", True)
        if not (t.get_bit("triggerable") or t.get_bit("save_point")):
            t.set_bit("encounters", True)

    available_placements = PlacementObject.available_placements
    PlacementObject.canonical_zero = available_placements[0]
    for m in MapObject.every:
        m.set_bit("magnetic", False)
        m.set_bit("exitable", False)
        m.set_bit("warpable", False)
        if len(m.npc_placements) == 0:
            m.npc_placement_index = PlacementObject.canonical_zero
        EncounterRateObject.get(m.index).encounter_rate = 1 + sum(
            [random.randint(0, 3) for _ in xrange(3)])

    for i in InitialEquipObject.every:
        for attr, _, _ in InitialEquipObject.specsattrs:
            setattr(i, attr, 0)

    CommandObject.get(2).commands = CommandObject.get(0x10).commands
    event_spells = [
        (0x1d, 3), (0x1e, 3), (0x21, 3), (0x24, 3),
        (0x16, 7), (0x42, 0xc), (0x43, 0xc), (0x44, 0xc),
        ]
    for spell, groupindex in event_spells:
        InitialSpellObject.add_spell(spell, groupindex)

    for i in InitialSpellObject.getgroup(10):
        i.groupindex = -1
    for i in InitialSpellObject.getgroup(11):
        i.groupindex = -1

    cluster_groups, lungs = generate_cave_layout()
    mapid, x, y = cluster_groups[0].home
    setup_opening_event(mapid=mapid, x=x, y=y)
    write_location_names()


def undummy():
    for line in open(path.join(tblpath, "dummied_items.txt")):
        line = line.strip('\n')
        args = line.split("|")
        index = int(args[0], 0x10)
        name = args[1:]
        i = ItemNameObject.get(index)
        i.rename(name)

    for line in open(path.join(tblpath, "dummied_commands.txt")):
        line = line.strip('\n')
        index, name, job, position = line.split()
        try:
            index, job, position = (
                int(index, 0x10), int(job, 0x10), int(position))
        except ValueError:
            continue
        name = name.split('|')
        c = CommandNameObject.get(index)
        c.rename(name)
        c = CommandObject.get(job)
        c.commands.insert(position, index)
        c.commands = c.commands[:5]

    for line in open(path.join(tblpath, "dummied_spells.txt")):
        line = line.strip('\n')
        args = line.split("|")
        index = int(args[0], 0x10)
        name = args[1:]
        try:
            s = SpellNameObject.get(index)
        except KeyError:
            s = LongSpellNameObject.get(index-len(SpellNameObject.every))
        s.rename(name)

    rosa_learn = [(5, 12), (6, 27), (0xc, 31)]
    rosa_spellset = 7
    for spell, level in rosa_learn:
        LearnedSpellObject.add_spell(spell, level, rosa_spellset)


def lunar_ai_fix(filename=None):
    if filename is None:
        filename = get_outfile()
    FORMATION_FLAGS_ADDR = addresses.lunar_formation_flags
    FUNCTION_ADDR = addresses.lunar_ai_function
    OVERWRITE_LIMIT = addresses.lunar_ai_limit

    function_asm = [
        0xDA,                   # PHX
        0x5A,                   # PHY
        0xAD, 0x00, 0x18,       # LDA $1800
        0xAA,                   # TAX
        0x29, 0x07,             # AND #$07
        0xA8,                   # TAY
        0x8A,                   # TXA
        0x4A,                   # LSR A
        0x4A,                   # LSR A
        0x4A,                   # LSR A
        0xAA,                   # TAX
        0xA9, 0x00,             # LDA #$00
        0xC8,                   # INY
        0x38,                   # SEC
        0x2A,                   # ROL A
        0x88,                   # DEY
        0xD0, 0xFC,             # BNE -2
        # AND $7D40,X
        0x3F, FORMATION_FLAGS_ADDR & 0xFF, (FORMATION_FLAGS_ADDR >> 8) & 0xFF, (FORMATION_FLAGS_ADDR >> 16),
        0xF0, 0x08,             # BEQ +8
        0xAD, 0x01, 0x18,       # LDA $1801
        0x09, 0x80,             # ORA $#80
        0x8D, 0x01, 0x18,       # STA $1801
        0x7A,                   # PLY
        0xFA,                   # PLX
        ]
    assert FUNCTION_ADDR + len(function_asm) <= OVERWRITE_LIMIT
    while FUNCTION_ADDR + len(function_asm) < OVERWRITE_LIMIT:
        function_asm.append(0xEA)

    f = open(filename, "r+b")
    f.seek(FUNCTION_ADDR)
    f.write("".join(map(chr, function_asm)))
    f.close()

    for f in FormationObject.every:
        f.is_lunar_ai_formation


def unload_fusoya_spellsets():
    for ss in InitialSpellObject.getgroup(11):
        ss.groupindex = -1


if __name__ == "__main__":
    try:
        print ('You are using the FF4 '
               'randomizer version %s.' % VERSION)
        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]
        run_interface(ALL_OBJECTS, snes=True)
        get_location_names()
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))

        if get_global_label() == "FF2_US_11":
            undummy()
        lunar_ai_fix()
        setup_cave()
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF4-R", VERSION, lorom=True)
        finish_interface()
    except IOError, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
