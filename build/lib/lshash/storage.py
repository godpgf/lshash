# lshash/storage.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import json

try:
    import redis
except ImportError:
    redis = None

__all__ = ['storage']


def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    elif 'redis' in storage_config:
        storage_config['redis']['db'] = index
        return RedisStorage(storage_config['redis'])
    else:
        raise ValueError("Only in-memory dictionary and Redis are supported.")


class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def keys(self, binary_hash=None, max_hamming_dist=2):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class KeyNode(object):
    def __init__(self, value=None):
        self.left = None
        self.right = None
        self.pre = None
        self.value = value

    def get_value(self):
        value_list = []
        node = self
        while node.pre is not None:
            if node.value is not None:
                value_list.append(node.value)
            node = node.pre
        value_list.reverse()
        return "".join(value_list)


class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()
        self.key_root = KeyNode()

    def keys(self, binary_hash=None, max_hamming_dist=2):
        if binary_hash is None:
            return self.storage.keys()
        else:
            key_list = []
            self.search_keys(key_list, list(binary_hash), 0, max_hamming_dist, self.key_root)
            return key_list


    @classmethod
    def search_keys(cls, key_list, binary_hash_list, cur_id, remain_hamming_dist, root):
        if cur_id == len(binary_hash_list):
            key_list.append(root.get_value())
        else:
            if root.left is not None:
                if binary_hash_list[cur_id] == '0':
                    cls.search_keys(key_list, binary_hash_list, cur_id+1, remain_hamming_dist, root.left)
                elif remain_hamming_dist > 0:
                    cls.search_keys(key_list, binary_hash_list, cur_id+1, remain_hamming_dist-1, root.left)
            if root.right is not None:
                if binary_hash_list[cur_id] == '1':
                    cls.search_keys(key_list, binary_hash_list, cur_id+1, remain_hamming_dist, root.right)
                elif remain_hamming_dist > 0:
                    cls.search_keys(key_list, binary_hash_list, cur_id+1, remain_hamming_dist-1, root.right)

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)
        key_list = list(key)
        cur_index = 0
        pre_node = self.key_root
        while cur_index < len(key_list):
            if key_list[cur_index] == '0':
                if pre_node.left is None:
                    pre_node.left = KeyNode('0')
                pre_node = pre_node.left
                cur_index += 1
            elif key_list[cur_index] == '1':
                if pre_node.right is None:
                    pre_node.right = KeyNode('1')
                pre_node = pre_node.right
                cur_index += 1
            else:
                break

    def get_list(self, key):
        return self.storage.get(key, [])


class RedisStorage(BaseStorage):
    def __init__(self, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.name = 'redis'
        self.storage = redis.StrictRedis(**config)

    def keys(self, pattern="*"):
        return self.storage.keys(pattern)

    def set_val(self, key, val):
        self.storage.set(key, val)

    def get_val(self, key):
        return self.storage.get(key)

    def append_val(self, key, val):
        self.storage.rpush(key, json.dumps(val))

    def get_list(self, key):
        return self.storage.lrange(key, 0, -1)
