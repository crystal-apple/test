#给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

def solution(nums,target):
    tar = []
    if len(nums)<2:
        return
    for i in range(0,len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i] + nums[j] == target:
                tar.append([i,j])
    return tar
if __name__=='__main__':
    nums = [2, 7, 3, 6, 11, 15]
    target = 9
    print(solution(nums,target))
        
