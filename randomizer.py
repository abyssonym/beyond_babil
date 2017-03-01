from randomtools.tablereader import TableObject, get_global_label, tblpath
from randomtools.utils import (
    classproperty, mutate_normal, shuffle_bits, get_snes_palette_transformer,
    utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, run_interface, rewrite_snes_meta,
    clean_and_write, finish_interface)
from collections import defaultdict
from os import path
from time import time
import string


VERSION = 1
ALL_OBJECTS = None
LOCATION_NAMES = []


def get_location_names(outfile):
    if LOCATION_NAMES:
        return LOCATION_NAMES

    f = open(outfile, "r+b")
    f.seek(0xa9620)
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
    print hex(f.tell())
    f.close()
    return get_location_names(None)


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

    def __repr__(self):
        return "%x: %s %s" % (self.cluster.mapid, self.x, self.y)

    def is_adjacent(self, other):
        return (self.x-other.x) * (self.y-other.y) == 0 and (
            abs(self.x-other.x) == 1 or abs(self.y-other.y) == 1)


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
    def exit_adjacencies(self):
        if hasattr(self, "_exit_adjacencies"):
            return self._exit_adjacencies
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

        self._exit_adjacencies = sorted(adjacencies)
        return self.exit_adjacencies

    @property
    def num_outs(self):
        return len(self.exit_adjacencies)


class ClusterGroup:
    def __init__(self, clusters):
        self.clusters = sorted(set(clusters))

    def connect(self):
        self.connections = []
        edge_candidates = [c for c in self.clusters if c.num_outs >= 2]
        if len(edge_candidates) < 2:
            raise Exception("Not enough doors for two entrances to group")

        start, finish = random.sample(edge_candidates, 2)
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
                    raise Exception("Unable to connect all clusters.")
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
            raise Exception("Unable to connect all clusters.")
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

    def full_execute(self):
        for _ in xrange(10):
            try:
                self.connect()
                break
            except:
                pass
        else:
            raise Exception("Excessive attempts to connect clusters.")
        self.rank_clusters()
        self.fill_out()


def try_it_out():
    clusterpath = path.join(tblpath, "clusters.txt")
    f = open(clusterpath)
    clusters = [Cluster.from_string(line) for line in f.readlines()
                if line.strip()]
    f.close()
    random.seed(int(time()))
    while True:
        try:
            cg = ClusterGroup(random.sample(clusters, 20))
            cg.full_execute()
            break
        except:
            continue
    print cg.start.rank, cg.start
    print
    for a, b in cg.connections:
        print a.rank, a
        if b is not None:
            print b.rank, b
        else:
            print b
        print
    print cg.finish.rank, cg.finish
    print
    import pdb; pdb.set_trace()


class FormationObject(TableObject):
    def __repr__(self):
        names = []
        for i, t in enumerate(self.monster_types):
            try:
                m = MonsterNameObject.get(t)
            except KeyError:
                continue
            num = (self.monster_qty >> (6 - (2*i))) & 0b11
            names.append("%s x%s" % (m.name, num))
        return "%x %s" % (self.index, ", ".join(names))
        return "%x %s %s %s" % (self.index, self.rank, self.prerank, ", ".join(names))

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
    def prerank(self):
        if hasattr(self, "_prerank"):
            return self._prerank
        rank = 0
        monsters = sorted(self.monsters, key=lambda m: m.rank, reverse=True)
        assert monsters[0].rank >= monsters[-1].rank
        for i, m in enumerate(monsters):
            rank += (m.rank * (0.5 ** i))
        if self.get_bit("back_attack"):
            rank *= (1 + (0.1 * len(monsters)))
        #if self.get_bit("no_flee"):
        #    rank *= 1.5
        if self.get_bit("egg1") or self.get_bit("egg2") or self.get_bit("egg3"):
            rank *= 0.5
        if self.get_bit("no_gameover"):
            rank = 0
        if self.get_bit("character_battle") or self.get_bit("auto_battle"):
            rank = -1
        self._prerank = rank
        return self.prerank

    @property
    def rank(self):
        return self.prerank

        if hasattr(self, "_rank"):
            return self._rank
        if not hasattr(self, "_prerank"):
            [f.prerank for f in FormationObject.every]
        is_boss = lambda f: f.get_bit("boss_death") and f.get_bit("no_flee")
        boss_status = is_boss(self)
        imp_x1 = FormationObject.get(0x1FF)
        like_formations = [f for f in FormationObject.every if f.prerank > imp_x1.prerank and is_boss(f) == boss_status]
        if self not in like_formations:
            self._rank = -1
            return self.rank
        index = sorted(like_formations, key=lambda f: f.prerank).index(self)
        self._rank = int(round(index * 1000 / float(len(like_formations))))
        return self.rank

    @property
    def battle_music(self):
        return (self.misc2 >> 2) & 0b11


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
        return s.strip()


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

    @property
    def name(self):
        return MonsterNameObject.get(self.index).name

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
        rank = (self.get_attr_rank("hp") +
                (self.get_attr_rank("xp") * 10))
        return rank * 1000000

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


class EncounterObject(TableObject): pass
class PackObject(TableObject): pass
class ItemNameObject(NameObject): pass
class CommandNameObject(NameObject): pass


class EventObject(TableObject):
    BASE_POINTER = 0x90200

    @property
    def instructions(self):
        if hasattr(self, "_instructions"):
            return self._instructions
        f = open(get_outfile(), "r+b")
        f.seek(self.event_pointer + self.BASE_POINTER)
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
    def battle(self):
        return 0xEC in self.commands

    @property
    def flagger(self):
        return bool(set(self.commands) & set([0xF2, 0xF3]))


class EventCallObject(TableObject):
    BASE_POINTER = 0x97460

    @property
    def size(self):
        try:
            return (self.get(self.index+1).event_call_pointer
                - self.event_call_pointer)
        except KeyError:
            return 0

    @property
    def cases(self):
        if hasattr(self, "_cases"):
            return self._cases
        self._cases = []

        f = open(get_outfile(), "r+b")
        f.seek(self.event_call_pointer + self.BASE_POINTER)
        conditions = []
        while True:
            if f.tell() >= self.BASE_POINTER + self.event_call_pointer + self.size:
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


class PlacementObject(TableObject):
    def neutralize(self):
        self.set_bit("intangible", True)
        self.set_bit("walks", False)
        self.npc_index = 0

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
        return SpeechObject.get(self.npc_index)

    @property
    def events(self):
        return self.speech.events

    @property
    def messager(self):
        return any([e.messager for e in self.events])


class SpeechObject(EventCallObject):
    BASE_POINTER = 0x99c00


class TileObject(TableObject):
    @property
    def walkable(self):
        return (self.get_bit("layer_1") or self.get_bit("layer_2")) and not self.get_bit("bridge_layer")


class TriggerObject(TableObject):
    # x, y are not modifiable without changing the map data
    # you just end up on the moon

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
        for i, (x, y) in enumerate(self.warps):
            s += "WARP {0:0>2}: {1:0>2} {2:0>2}\n".format(i, x, y)
        return s.strip()

    @property
    def name(self):
        #if self.name_index >= len(LOCATION_NAMES):
        #    return "?????"
        return "%x %s" % (self.index, LOCATION_NAMES[self.name_index & 0x7F])

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
    def chests(self):
        return [c for c in self.triggers if c.is_chest]

    @property
    def events(self):
        return [e for e in self.triggers if e.is_event]

    @property
    def npc_placements(self):
        return PlacementObject.getgroup(self.npc_placement_index)

    @property
    def exit_summary(self):
        summary = set([])
        for e in self.exits:
            summary.add((e.x, e.y))
        for x, y in self.warps:
            summary.add((x, y))
        return summary


class MapGridObject(TableObject):
    BASE_POINTER = 0xb8000

    @property
    def map(self):
        if hasattr(self, "_map"):
            return self._map

        self._compressed = ""
        f = open(get_outfile(), "r+b")
        f.seek(self.map_pointer + self.BASE_POINTER)
        data = []
        while len(data) < 1024:
            tile = ord(f.read(1))
            self._compressed += chr(tile)
            if tile & 0x80:
                tile = tile & 0x7F
                data.append(tile)
                additional = ord(f.read(1))
                self._compressed += chr(tile)
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


class MapGrid2Object(MapGridObject):
    BASE_POINTER = 0xc0000


if __name__ == "__main__":
    try:
        print ('You are using the FF4 '
               'randomizer version %s.' % VERSION)
        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]
        run_interface(ALL_OBJECTS, snes=True)
        get_location_names(get_outfile())
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))
        #for f in sorted(FormationObject.every):
        #    if f.get_bit("no_flee") and f.battle_music in [1, 2, 3]:
        #        print f

        #for m in MonsterObject.every:
        #    print m.name, m.level & 0x7F, m.hp, m.attack, m.defense, m.magic_defense, "%x" % m.drop_speed, m.xp, m.gil
        #for m in MonsterObject.ranked:
        #    print m, m.hp, m.xp, m.defense, m.magic_defense
        from collections import Counter
        used_tilesets = sorted(Counter(m.tileset_index for m in MapObject.every).items(), key=lambda (a, b): b)
        '''
        for f in FormationObject.every:
            if "Zeromus" in str(f):
                print f
                print
        for e in EventObject.every:
            print e.pretty_script
            print
        flags = set([])
        for p in SpeechObject.every + EventCallObject.every:
            for conditions, call in p.cases:
                for flag, truth in conditions:
                    flags.add(flag)
        print [f for f in xrange(0x100) if f not in flags]
        crash_game = EventObject.get(0xFE)
        for t in TriggerObject.every:
            assert crash_game not in t.events
        for t in TriggerObject.every + PlacementObject.every:
            for e in t.events:
                if e.flagger:
                    t.neutralize()
                    break
            else:
                if crash_game in t.events and isinstance(t, PlacementObject):
                    t.set_bit("intangible", True)
        '''
        try_it_out()
        import pdb; pdb.set_trace()
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF4-R", VERSION, lorom=True)
        finish_interface()
    except IOError, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
