from observer import controller_cython as controller

import dataset

db = dataset.connect('sqlite:////home/tubuntu/data/barsave.db')

instab = db['instruments']

cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')

for ins in cont.allowed_ins:
    dbobj = { 'name': ins.name,
              'type': ins.type,
              'displayName': ins.displayName,
              'pipLocation': ins.pipLocation }
    instab.insert(dbobj)
