import { useState } from 'react'
import { useApp } from '../context/AppContext'
import './ProfileStep.css'

export default function ProfileStep() {
  const { setUserProfile } = useApp()
  const [name, setName] = useState('')
  const [age, setAge] = useState('')

  const valid = name.trim().length > 0 && Number(age) > 0 && Number(age) < 150

  function handleSubmit(e) {
    e.preventDefault()
    if (!valid) return
    setUserProfile({ name: name.trim(), age: Number(age) })
  }

  return (
    <div className="profile-step">
      <div className="profile-card">
        <div className="profile-header">
          <div className="profile-icon">🧠</div>
          <h2 className="profile-title">측정 시작 전 정보 입력</h2>
          <p className="profile-subtitle">입력하신 정보는 세션 레포트에 사용됩니다.</p>
        </div>

        <form className="profile-form" onSubmit={handleSubmit}>
          <div className="profile-field">
            <label className="profile-label" htmlFor="pf-name">이름</label>
            <input
              id="pf-name"
              className="profile-input"
              type="text"
              placeholder="이름을 입력하세요"
              value={name}
              onChange={e => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="profile-field">
            <label className="profile-label" htmlFor="pf-age">나이</label>
            <input
              id="pf-age"
              className="profile-input"
              type="number"
              placeholder="나이를 입력하세요"
              min="1"
              max="149"
              value={age}
              onChange={e => setAge(e.target.value)}
            />
          </div>

          <button
            type="submit"
            className="profile-submit"
            disabled={!valid}
          >
            시작하기 ▶
          </button>
        </form>
      </div>
    </div>
  )
}
