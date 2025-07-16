import 'package:security_app/features/monitoring/data/services/monitor_service.dart';

class MonitoringRepository {
  final MonitoringService service;

  MonitoringRepository(this.service);

  Future<bool> checkInternetConnection() {
    return service.checkInternetConnection();
  }

  Future<bool> checkPermissions() {
    return service.checkPermissions();
  }

  Future<bool> isVideoFeedAvailable() {
    return service.isVideoFeedAvailable();
  }
}
