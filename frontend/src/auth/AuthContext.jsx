import { createContext, useContext, useEffect, useState } from 'react'
import { getAuthToken, setAuthToken, clearAuthToken, login as apiLogin, signup as apiSignup, getCurrentUser } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => getAuthToken())
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(!!token)

  useEffect(() => {
    if (!token) {
      setUser(null)
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    getCurrentUser()
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .catch(() => {
        if (!cancelled) {
          clearAuthToken()
          setToken(null)
          setUser(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [token])

  const login = async (email, password) => {
    const res = await apiLogin(email, password)
    if (res?.access_token) {
      setAuthToken(res.access_token)
      setToken(res.access_token)
      const me = await getCurrentUser()
      setUser(me)
    }
  }

  const signup = async (email, password) => {
    await apiSignup(email, password)
    await login(email, password)
  }

  const logout = () => {
    clearAuthToken()
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return ctx
}

