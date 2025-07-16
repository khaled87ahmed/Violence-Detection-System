abstract class MonitoringState {}

class MonitoringInitial extends MonitoringState {}

class MonitoringLoading extends MonitoringState {}

class MonitoringSuccess extends MonitoringState {}

class MonitoringError extends MonitoringState {
  final String message;

  MonitoringError({required this.message});
}
