const MetricsDisplay = ({ activitiesCount, sessionTime, nextCapture, formatTime }) => {
  const metrics = [
    {
      id: 'activities',
      label: 'Activities Captured',
      value: activitiesCount,
      valueClass: 'text-2xl font-semibold text-primary-obsidian',
      rotation: 'rotate-1'
    },
    {
      id: 'session',
      label: 'Session Duration', 
      value: formatTime(sessionTime),
      valueClass: 'text-2xl font-light font-mono text-primary-obsidian',
      rotation: '-rotate-1'
    },
    {
      id: 'next',
      label: 'Next Analysis',
      value: nextCapture !== null ? `${nextCapture}s` : '--',
      valueClass: 'text-lg font-medium text-accents-frost',
      rotation: 'rotate-0.5'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {metrics.map((metric, index) => (
        <div
          key={metric.id}
          className={`bg-primary-ghost border border-primary-slate/30 
                     rounded-lg p-6 text-center
                     transition-all duration-200 ease-out
                     hover:-translate-y-1 hover:shadow-lg hover:scale-105
                     ${metric.rotation}
                     animate-fade-in`}
          style={{
            animationDelay: `${index * 50}ms`,
            animationFillMode: 'both'
          }}
        >
          <div className={metric.valueClass}>
            {metric.value}
          </div>
          <div className="text-sm font-normal text-primary-charcoal mt-1">
            {metric.label}
          </div>
        </div>
      ))}
    </div>
  )
}

export default MetricsDisplay