#include <iostream>
#include <lua.hpp>
#include <glog/logging.h>

// CHECK()s if a Lua method returned an error.
void CheckForLuaErrors(lua_State* L, int status) {
    CHECK_EQ(status, 0) << lua_tostring(L, -1);
}

class LuaParameterDictionary {
public:
    LuaParameterDictionary(const std::string& code): L_(luaL_newstate()) {
        luaL_openlibs(L_);
        CheckForLuaErrors(L_, luaL_loadstring(L_, code.c_str()));
        CheckForLuaErrors(L_, lua_pcall(L_, 0, 1, 0));
    }
    int GetInt(const std::string& key) {
        CheckHasKey(key);
        GetValueFromLuaTable(L_, key);
        return PopInt();
    }

    void CheckHasKey(const std::string& key) const {
        CHECK(HasKey(key)) << "Key '" << key << "' not in dictionary:\n";
    }
    bool HasKey(const std::string& key) const {
        return HasKeyOfType(L_, key);
    }
    // Returns true if 'key' is in the table at the top of the Lua stack.
    template <typename T>
    static bool HasKeyOfType(lua_State* L, const T& key) {
        PushValue(L, key);
        lua_rawget(L, -2);
        const bool key_not_found = lua_isnil(L, -1);
        lua_pop(L, 1);
        return !key_not_found;
    }
    static void PushValue(lua_State* L, const int key) { lua_pushinteger(L, key); }
    static void PushValue(lua_State* L, const std::string& key) {
        lua_pushstring(L, key.c_str());
    }
    template <typename T>
    static void GetValueFromLuaTable(lua_State* L, const T& key) {
        PushValue(L, key);
        lua_rawget(L, -2);
    }
    int PopInt() const {
        CHECK(lua_isnumber(L_, -1)) << "Top of stack is not a number value.";
        const int value = lua_tointeger(L_, -1);
        lua_pop(L_, 1);
        return value;
    }
private:
    lua_State* L_;  // The name is by convention in the Lua World.
};
std::string PrintType(int value_type) {
    switch (value_type) {
        case LUA_TBOOLEAN:
            return "Bool";
            break;
        case LUA_TSTRING:
            return "String";
            break;
        case LUA_TNUMBER: {
            return "Number";
        } break;
        case LUA_TTABLE: {
            return "Table";
        } break;
        case LUA_TFUNCTION: {
            return "Function";
        } break;
        default:
            return "UNkonwn";
    }
}
void test1() {
    std::string code = "return { age = 10, hour = 60 }";
    lua_State* L_ = luaL_newstate();    // create new one
    luaL_openlibs(L_);
    CheckForLuaErrors(L_, luaL_loadstring(L_, code.c_str()));  // load lua code
    LOG(INFO) << "size=" << lua_gettop(L_) << " top="           // lua code is a function object, size=1 now
              << PrintType(lua_type(L_, -1));
    CheckForLuaErrors(L_, lua_pcall(L_, 0, 1, 0));             // call it
    LOG(INFO) << "size=" << lua_gettop(L_) << " top="          // clean function object, get a table, size=1 now
              << PrintType(lua_type(L_, -1));
    lua_pushstring(L_, "hour");                                // push string
    LOG(INFO) << "size=" << lua_gettop(L_) << " top="          // size=2, top is string
              << PrintType(lua_type(L_, -1));
    lua_rawget(L_, -2);                                        // get table[key], clean key, size now is 2
    LOG(INFO) << "size=" << lua_gettop(L_) << " top="
              << PrintType(lua_type(L_, -1));
}
int main() {
    test1();
    LuaParameterDictionary dir("return { age = 10, hour = 60 }");
    LOG(INFO) << "age:" << dir.GetInt("age");
    LOG(INFO) << "age:" << dir.GetInt("age");
    LOG(INFO) << "hour:" << dir.GetInt("hour");
    return 0;
}
