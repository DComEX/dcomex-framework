import follow


@follow.follow()
def a(i):
    if i > 0:
        b(i - 1)


@follow.follow()
def b(i):
    c(i)


@follow.follow()
def c(i):
    a(i)


a(3)

print("has loop:", follow.loop())
with open("follow1.gv", "w") as file:
    follow.graphviz(file)
