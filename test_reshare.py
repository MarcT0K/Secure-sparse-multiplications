from mpyc import thresha
from mpyc.runtime import mpc

secint = mpc.SecInt(64)
m = 3
t = 1

VAL = 3

x = secint(VAL)
field = x.field

in_shares = thresha.random_split(field, [x.share], t, m)
print("X=", VAL)
print("in_shares=", in_shares)

shares_to_points = lambda shares: [(j + 1, s) for j, s in enumerate(shares)]

z = thresha.recombine(field, shares_to_points(in_shares))
assert z[0] % field.modulus == VAL

share1 = in_shares[1]
share2 = in_shares[2]

in_shares1 = thresha.random_split(field, share1, t, m)
in_shares2 = thresha.random_split(field, share2, t, m)

party1 = [in_shares1[0], in_shares2[0]]
party2 = [in_shares1[1], in_shares2[1]]
party3 = [in_shares1[2], in_shares2[2]]


sp1 = thresha.recombine(
    field, [(2, party1[0]), (3, party1[1])]
)  # We recombine with the position of the shared shares
sp2 = thresha.recombine(field, [(2, party2[0]), (3, party2[1])])
sp3 = thresha.recombine(field, [(2, party3[0]), (3, party3[1])])

sp = sp = [sp1, sp2, sp3]

z = thresha.recombine(field, shares_to_points(sp))
assert z[0] % field.modulus == VAL
print("z=", z)
