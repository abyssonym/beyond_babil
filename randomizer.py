from randomtools.tablereader import TableObject, get_global_label, tblpath
from randomtools.utils import (
    classproperty, mutate_normal, shuffle_bits, get_snes_palette_transformer,
    utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, run_interface, rewrite_snes_meta,
    clean_and_write, finish_interface)
from collections import defaultdict
from os import path
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
    @property
    def instructions(self):
        if hasattr(self, "_instructions"):
            return self._instructions
        f = open(get_outfile(), "r+b")
        f.seek(self.event_pointer | 0x90200)
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

class PlacementUnit:
    def __init__(self, npc_index, x, y, misc):
        self.npc_index = npc_index
        self._x = x
        self._y = y
        self.misc = misc

    @property
    def x(self):
        return self._x & 0x1F

    @property
    def y(self):
        return self._y & 0x1F

    @property
    def walks(self):
        return bool(self.x & 0x80)

    @property
    def intangible(self):
        return bool(self.y & 0x80)

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


class PlacementObject(TableObject):
    pass


class TileObject(TableObject):
    @property
    def walkable(self):
        return (self.get_bit("layer_1") or self.get_bit("layer_2")) and not self.get_bit("bridge_layer")


class TriggerObject(TableObject):
    # x, y are not modifiable without changing the map data
    # you just end up on the moon

    @property
    def description(self):
        s = "{0:0>2} {1:0>2}".format(self.x, self.y)
        if self.is_event:
            s = "%s EVENT: $%x" % (s, self.event)
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
    def event(self):
        return self.misc2

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
        return s.strip()

    @property
    def name(self):
        #if self.name_index >= len(LOCATION_NAMES):
        #    return "?????"
        return LOCATION_NAMES[self.name_index & 0x7F]

    @property
    def map(self):
        return MapGridObject.get(self.grid_index).map

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


class MapGridObject(TableObject):
    @property
    def map(self):
        if hasattr(self, "_map"):
            return self._map

        f = open(get_outfile(), "r+b")
        f.seek(self.map_pointer | 0xb8000)
        data = []
        while len(data) < 1024:
            tile = ord(f.read(1))
            if tile & 0x80:
                tile = tile & 0x7F
                data.append(tile)
                additional = ord(f.read(1))
                data.extend([tile] * additional)
            else:
                data.append(tile)
        f.close()
        self._map = [data[(i*32):((i+1)*32)] for i in xrange(32)]
        assert [len(row) == 32 for row in self._map]
        assert len(self._map) == 32
        return self.map

    @property
    def pretty_map(self):
        s = ""
        for row in self.map:
            s += " ".join(["{0:0>2}".format("%x" % v) for v in row])
            s += "\n"
        return s.strip()


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
        '''
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF4-R", VERSION, lorom=True)
        finish_interface()
    except IOError, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
