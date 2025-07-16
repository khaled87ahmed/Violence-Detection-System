import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:security_app/constant.dart';
import 'package:security_app/core/utils/app_router.dart';
import 'package:security_app/features/Auth/busines_logic/auth_cubit/cubit/auth_cubit.dart';
import 'package:security_app/features/monitoring/data/repo/monitoring_repository.dart';
import 'package:security_app/features/monitoring/data/services/monitor_service.dart';
import 'package:security_app/features/monitoring/logic/cubit/monitor_cubit.dart';
import 'package:security_app/firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await FirebaseAuth.instance.setSettings(
      appVerificationDisabledForTesting: true); // افتكر شيله لو مش هتحتاجه

  // Monitor API Service & Repository

  runApp(SecurityApp());
}

class SecurityApp extends StatelessWidget {
  const SecurityApp({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider(create: (context) => AuthCubit()),
        BlocProvider(
          create: (context) => MonitoringCubit(
            monitoringRepository: MonitoringRepository(
              MonitoringService(),
            ),
          ),
        ),
      ],
      child: MaterialApp.router(
        routerConfig: AppRouter.router,
        debugShowCheckedModeBanner: false,
        theme: ThemeData.dark().copyWith(
          scaffoldBackgroundColor: kPrimaryColor,
          textTheme:
              GoogleFonts.montserratTextTheme(ThemeData.dark().textTheme),
        ),
      ),
    );
  }
}
