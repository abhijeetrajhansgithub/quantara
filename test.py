from quantara import Database, list_indexes

db = Database("test")

print(type(db))

print(list_indexes())