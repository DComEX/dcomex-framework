import follow


@follow.follow()
def a():
    b()


@follow.follow(label="B")
def b():
    c(0)


c = follow.follow(label="c")(lambda i: i, )
a()

print("has loop:", follow.loop())
with open("follow0.gv", "w") as file:
    follow.graphviz(file)
