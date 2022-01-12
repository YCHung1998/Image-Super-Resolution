'''
Accumulate calculate the average.
1. First reset the value.
2. Then update each steps.
3. Return the mean item.
'''


class AccAvg(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, avg, n):
        self.val += (avg * n)
        self.count += n

    def item(self):
        return self.val / self.count if self.count else 0

if __name__ == '__main__':
    tmp = AccAvg()
    print(tmp.item())
    tmp.update(10, 2)
    print(tmp.item())
    tmp.update(4, 7)
    print(tmp.item())
